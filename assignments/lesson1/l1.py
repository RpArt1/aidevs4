import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from common import LLMService, AssignmentService, get_logger
from assignments.assignment import Assignment

class Lesson1(Assignment):
    def __init__(self):
        self.log = get_logger(__name__)
        self.llm = LLMService(model="qwen/qwen3-coder-30b-a3b-instruct")
        self.assignment = AssignmentService()

    test_json = """[
    {
        "id": 22,
        "job": "Głównym celem tej pracy jest przygotowanie podopiecznych do aktywnego uczestnictwa w życiu społecznym i zawodowym. Przekazuje im nie tylko wiedzę teoretyczną, ale także ucząc praktycznych umiejętności i wartości. Pomaga odkrywać talenty i rozwijać pasje."
    },
    {
        "id": 235,
        "job": "Zajmuje się on utrzymaniem w należytym stanie technicznym wszelkich pojazdów i maszyn. Jego praca wymaga doskonałej znajomości zasad działania silników, układów hamulcowych i kierowniczych. Precyzja i skrupulatność to jego cechy charakterystyczne."
    },
    {
        "id": 264,
        "job": "Osoba ta zgłębia zasady rządzące funkcjonowaniem atomów, cząsteczek i większych struktur. Zajmuje się badaniem właściwości materiałów i ich interakcji. Jej prace pomagają rozwijać nowe technologie."
    },
    {
        "id": 316,
        "job": "Opracowuje zasady i metody przeprowadzania badań, które pozwalają lepiej zrozumieć otaczający nas świat. Jego praca ma na celu odkrywanie nowych praw i zależności. Wyniki jego badań mogą mieć ogromne znaczenie praktyczne."
    },
    {
        "id": 581,
        "job": "Ściga przestępców, dbając o to, by nikt nie pozostał bezkarny za popełnione czyny. Przeszukuje miejsca zdarzeń, zbiera dowody i składając zeznania w sądzie. Jego celem jest utrzymanie praworządności."
    }
    ]"""

    JOB_DESCRIPTIONS_SCHEMA = {
        "name": "job_descriptions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": ["string", "null"],
                                "description": "The unique identifier found in the text. Use null if not mentioned."
                            },
                            "tags": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["IT","transport","edukacja","medycyna","praca z ludźmi","praca z pojazdami","praca fizyczna"]
                                },
                                "description": "List of relevant tags. Must only contain values from the allowed list. Empty array if none apply."
                            }
                        },
                        "required": ["id", "tags"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }
    }

    CACHE_FILE = Path(__file__).parent / "classified_jobs.json"
    FILTERED_FILE = Path(__file__).parent / "filtered_jobs.json"

    def _classify_jobs(self):
        if self.CACHE_FILE.exists():
            self.log.info(f"Loading cached results from {self.CACHE_FILE}")
            return json.loads(self.CACHE_FILE.read_text(encoding="utf-8"))

        prompt_path = Path(__file__).parent / "prompt.md"
        system_prompt = prompt_path.read_text(encoding="utf-8")

        try:
            jobs = json.loads(self.FILTERED_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            self.log.warning(f"Could not load jobs from {self.FILTERED_FILE}: {e}.")
            return []

        if not jobs:
            self.log.warning("Jobs list is empty.")
            return []

        user_message = json.dumps(jobs, ensure_ascii=False)
        response = self.llm.chat_structured(
            messages=[{"role": "user", "content": user_message}],
            schema=self.JOB_DESCRIPTIONS_SCHEMA,
            system_prompt=system_prompt,
        )
        results = response["results"]

        self.CACHE_FILE.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.log.info(f"Saved results to {self.CACHE_FILE}")
        return results

    def solve(self):
        classified = self._classify_jobs()
        tags_by_id = {str(c["id"]): c["tags"] for c in classified}

        people = json.loads(self.FILTERED_FILE.read_text(encoding="utf-8"))

        answer = []
        for person in people:
            pid = str(person["id"])
            tags = tags_by_id.get(pid, [])
            if "transport" not in tags:
                continue
            born_year = int(person["birthDate"][:4])
            answer.append({
                "name": person["name"],
                "surname": person["surname"],
                "gender": person["gender"],
                "born": born_year,
                "city": person["birthplace"],
                "tags": tags,
            })

        self.log.info(f"Found {len(answer)} transport workers")
        for p in answer:
            self.log.info(f"  {p['name']} {p['surname']} ({p['born']}) - {p['tags']}")

        self.log.info(f"Sending answer: {answer}")
        result = self.assignment.send("people", answer)
        
        self.log.info(f"Response: {result}")
        return result