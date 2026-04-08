from dotenv import load_dotenv

from common.llm_service import LLMService
from common.langfuse_tracing import init_tracing, shutdown_tracing, trace_agent, trace_generation

load_dotenv()

# Boot the Langfuse client once at startup — same pattern as initTracing() in TS.
# If the env vars are missing it logs a warning and all trace_* calls become no-ops.
init_tracing()

llm = LLMService(model="openai/gpt-4o-mini")
messages = [{"role": "user", "content": "What did one snowman say to the other snowman?"}]

# trace_agent opens the top-level trace visible in the Langfuse dashboard.
# All nested observations (generations, tool calls) become children of this span.
with trace_agent("langfuse-connection-test", input=messages) as agent_obs:

    # trace_generation marks this block as an LLM call, enabling token/cost tracking.
    with trace_generation("hello-generation", model=llm.model, input=messages) as gen_obs:
        reply = llm.chat(messages)

        # Record the output and token counts on the generation span.
        if gen_obs:
            gen_obs.update(
                output=reply,
                usage={
                    "input": llm.last_usage.input_tokens,
                    "output": llm.last_usage.output_tokens,
                },
            )

    # Attach the final answer at the top-level trace too.
    if agent_obs:
        agent_obs.update(output=reply)

# Flush all buffered events before the process exits — equivalent to shutdownTracing().
shutdown_tracing()

print(reply)
print("\nCheck your Langfuse dashboard for the logged trace.")
