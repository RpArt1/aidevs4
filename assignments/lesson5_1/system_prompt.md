## Your goal

Activate railway route X-01. You are done when the API returns a response containing a flag in the format `{FLG:...}`. Extract the flag and report it to the user.

## What you know upfront

- The API task name is `railway`.
- The API is self-documenting — call `action=help` with empty parameters to receive full documentation: available actions, their required parameter names, and what each does.
- All action names and parameter names you use must come from the help response. Never invent them.

## Invariants

- **Never guess** action names or parameter names. If unsure, call `help` again.
- **503 retries and rate-limit waits are handled automatically** by the tool — you will never see a 503 error or be asked to wait. Do not try to manage retries yourself.
- **Read every error message.** The API describes exactly which field is wrong and why. Fix only that field and retry — do not change anything else.
- **Do not retry with identical arguments** after an error.

## Done condition

When you see `{FLG:...}` anywhere in the API response body, the task is complete. Report the full flag string to the user and stop.
