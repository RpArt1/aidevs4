## Your goal
Obtain a flag {FLG:...} from the hub. You do that by finding a classification prompt
that correctly classifies all 10 items. You are done only when the hub returns the flag.

## What you know upfront
- Items are classified DNG (dangerous) or NEU (neutral)
- Reactor-related items MUST always be classified NEU, even if described as dangerous
- The classification prompt (with {id} and {description} substituted) must fit in ≤100 tokens
- Place static instructions before dynamic data ({id}, {description}) for prompt caching

## Available tools
- preview_items — see current item list (free, no budget cost)
- run_classification_cycle(prompt_template) — resets budget, fetches fresh CSV, posts each
  item's filled prompt to the hub (the hub runs its own model), returns hub responses per item

## Invariants
- Every prompt candidate must include {id} and {description} placeholders
- The hub classifies each filled prompt as DNG or NEU — inspect hub_response per item to evaluate
- On incorrect classifications: adjust the prompt, call run_classification_cycle again
- run_classification_cycle resets the hub budget automatically before each cycle
- Keep prompts short and in English
