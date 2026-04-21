"""Vision subagent for the lesson 2_2 electricity puzzle.

Pure helper — not a tool the planner sees. Given the raw board PNG, splits it
into a 3x3 grid with Pillow and asks a vision-capable model to describe each
tile independently (open edges + power plant marker). The planner agent only
ever sees the resulting structured JSON, never raw pixels.
"""

from __future__ import annotations

import base64
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any

from PIL import Image

from common import get_logger
from common.llm_service import LLMService


@dataclass(frozen=True)
class TileDescription:
    row: int
    col: int
    top: bool
    right: bool
    bottom: bool
    left: bool
    is_powerplant: bool
    label: str | None


class BoardVision:
    """Turn a raw electricity-board PNG into a structured 3x3 description.

    The board is split into 9 equal tiles using Pillow. Each tile is sent to the
    vision model in its own `chat_structured` call with a strict JSON schema,
    so the model has to return one reliable boolean-per-edge answer per tile
    with no cross-tile confusion.
    """

    GRID_SIZE = 3
    DEFAULT_MODEL = "google/gemini-3-flash-preview"
    DEFAULT_SAMPLES = 3

    _TILE_SCHEMA: dict[str, Any] = {
        "name": "electricity_tile",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "top": {
                    "type": "boolean",
                    "description": "True if a cable terminates at the top edge of this tile.",
                },
                "right": {
                    "type": "boolean",
                    "description": "True if a cable terminates at the right edge of this tile.",
                },
                "bottom": {
                    "type": "boolean",
                    "description": "True if a cable terminates at the bottom edge of this tile.",
                },
                "left": {
                    "type": "boolean",
                    "description": "True if a cable terminates at the left edge of this tile.",
                },
                "is_powerplant": {
                    "type": "boolean",
                    "description": "True if the tile shows a power plant icon (building + label).",
                },
                "label": {
                    "type": ["string", "null"],
                    "description": (
                        "Power plant label visible in the tile (e.g. 'PWR1593PL', "
                        "'PWR6132PL', 'PWR7264PL') or null if no label is present."
                    ),
                },
            },
            "required": ["top", "right", "bottom", "left", "is_powerplant", "label"],
            "additionalProperties": False,
        },
    }

    _TILE_PROMPT = (
        "You are looking at one single square tile cut out of a 3x3 cable-puzzle board. "
        "Examine ONLY this tile — ignore anything you think might belong to neighbouring tiles. "
        "For each of the four edges (top, right, bottom, left), report whether a cable "
        "reaches and terminates at that edge. A cable 'terminates' at an edge when its "
        "end touches the tile border. If the tile is empty or the cable does not reach "
        "the border, that edge is false.\n\n"
        "Also report whether the tile contains a power plant icon (a small building/tower "
        "marker with a label). If a label is visible (e.g. 'PWR1593PL', 'PWR6132PL', "
        "'PWR7264PL'), return it verbatim in 'label'; otherwise 'label' is null."
    )

    def __init__(
        self,
        llm: LLMService | None = None,
        model: str = DEFAULT_MODEL,
        samples: int = DEFAULT_SAMPLES,
    ) -> None:
        self._llm = llm if llm is not None else LLMService(model=model)
        self._log = get_logger(__name__)
        # Per-tile sample count for majority voting. The vision model is noisy
        # at the per-tile level (edges flicker, OCR labels drift between reads
        # of the same image), so we sample N independent times and vote each
        # field. Set samples=1 to disable voting.
        self._samples = max(1, int(samples))

    def describe_board(self, png_bytes: bytes) -> dict[str, Any]:
        """Split ``png_bytes`` into a 3x3 grid and describe each tile.

        Returns a dict shaped as::

            {
                "grid": [
                    [ {row, col, top, right, bottom, left, is_powerplant, label}, x3 ],
                    ... 3 rows ...
                ],
                "board_size": {"width": int, "height": int},
                "tile_size": {"width": int, "height": int},
            }

        Row/column indices are 1-based to match the hub's ``AxB`` cell addresses
        (``cell = f"{col}x{row}"``), so a planner reading the grid can wire
        directly into the ``rotate`` tool.
        """
        tiles, board_size, tile_size = self._split(png_bytes)

        grid: list[list[dict[str, Any]]] = []
        for row_idx, row_tiles in enumerate(tiles, start=1):
            row_descriptions: list[dict[str, Any]] = []
            for col_idx, tile_png in enumerate(row_tiles, start=1):
                description = self._describe_tile(tile_png, row=row_idx, col=col_idx)
                row_descriptions.append(asdict(description))
            grid.append(row_descriptions)

        return {
            "grid": grid,
            "board_size": {"width": board_size[0], "height": board_size[1]},
            "tile_size": {"width": tile_size[0], "height": tile_size[1]},
        }

    def _split(
        self, png_bytes: bytes
    ) -> tuple[list[list[bytes]], tuple[int, int], tuple[int, int]]:
        """Crop the board into a GRID_SIZE x GRID_SIZE list of PNG-encoded tiles."""
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        width, height = image.size
        tile_w = width // self.GRID_SIZE
        tile_h = height // self.GRID_SIZE

        if tile_w == 0 or tile_h == 0:
            raise ValueError(
                f"Board image too small to split into {self.GRID_SIZE}x{self.GRID_SIZE}: "
                f"{width}x{height}"
            )

        if width % self.GRID_SIZE or height % self.GRID_SIZE:
            self._log.warning(
                "board dimensions %dx%d not evenly divisible by %d — "
                "tiles will drop %d trailing px horizontally and %d vertically",
                width,
                height,
                self.GRID_SIZE,
                width % self.GRID_SIZE,
                height % self.GRID_SIZE,
            )

        self._log.info(
            "split board size=%dx%d tile=%dx%d", width, height, tile_w, tile_h
        )

        tiles: list[list[bytes]] = []
        for row in range(self.GRID_SIZE):
            row_tiles: list[bytes] = []
            for col in range(self.GRID_SIZE):
                box = (
                    col * tile_w,
                    row * tile_h,
                    col * tile_w + tile_w,
                    row * tile_h + tile_h,
                )
                tile_image = image.crop(box)
                buf = io.BytesIO()
                tile_image.save(buf, format="PNG")
                row_tiles.append(buf.getvalue())
            tiles.append(row_tiles)

        return tiles, (width, height), (tile_w, tile_h)

    def _describe_tile(self, tile_png: bytes, row: int, col: int) -> TileDescription:
        """Sample the vision model N times for one tile and majority-vote each field.

        The model is non-deterministic on this puzzle (edges and OCR labels
        drift between reads of the same image), so a single call is unreliable.
        We issue ``self._samples`` parallel calls and take the majority for
        each boolean field plus the most-common stripped label.
        """
        b64 = base64.b64encode(tile_png).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": self._TILE_PROMPT},
                ],
            }
        ]

        if self._samples == 1:
            payloads = [self._call_vision(messages, row, col)]
        else:
            with ThreadPoolExecutor(max_workers=self._samples) as pool:
                payloads = list(
                    pool.map(
                        lambda _i: self._call_vision(messages, row, col),
                        range(self._samples),
                    )
                )

        voted = self._majority_vote(payloads)

        description = TileDescription(
            row=row,
            col=col,
            top=voted["top"],
            right=voted["right"],
            bottom=voted["bottom"],
            left=voted["left"],
            is_powerplant=voted["is_powerplant"],
            label=voted["label"],
        )
        self._log.info(
            "tile row=%d col=%d edges=T%d/R%d/B%d/L%d powerplant=%s label=%s "
            "samples=%d",
            row,
            col,
            description.top,
            description.right,
            description.bottom,
            description.left,
            description.is_powerplant,
            description.label,
            self._samples,
        )
        return description

    def _call_vision(
        self, messages: list[dict[str, Any]], row: int, col: int
    ) -> dict[str, Any]:
        """One structured vision call for a tile. Raises on hard failure."""
        try:
            return self._llm.chat_structured(
                messages=messages, schema=self._TILE_SCHEMA
            )
        except Exception:
            self._log.exception("vision call failed row=%d col=%d", row, col)
            raise

    @staticmethod
    def _majority_vote(payloads: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine N tile payloads into one by majority-voting each field.

        Booleans: True iff strictly more than half the samples voted True
        (ties on an even sample count fall to False — conservative).
        Label: most-common non-empty stripped string; None if no string label
        appears in any sample, or if all samples disagree (no plurality).
        """
        n = len(payloads)
        threshold = n // 2  # majority = strictly more than half

        def vote_bool(key: str) -> bool:
            yes = sum(1 for p in payloads if bool(p.get(key)))
            return yes > threshold

        labels: list[str] = []
        for p in payloads:
            raw = p.get("label")
            if isinstance(raw, str):
                stripped = raw.strip()
                if stripped:
                    labels.append(stripped)

        label: str | None = None
        if labels:
            most_common, count = Counter(labels).most_common(1)[0]
            # Require at least 2 agreeing samples when N>=3, otherwise any
            # single noisy OCR hallucination would win for free.
            if n == 1 or count >= 2:
                label = most_common

        return {
            "top": vote_bool("top"),
            "right": vote_bool("right"),
            "bottom": vote_bool("bottom"),
            "left": vote_bool("left"),
            "is_powerplant": vote_bool("is_powerplant"),
            "label": label,
        }
