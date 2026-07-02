"""
main.py
-------
This is "User Query" and "Done" from the diagram. It collects input from
the terminal, hands it to the agent loop, and prints the final answer.
Every step (reasoning, actions, observations, parse errors) is also
written to agent.log for later inspection.
"""

import logging

from dotenv import load_dotenv

from agent import ReActAgent

load_dotenv()  # loads OPENAI_API_KEY / AGENT_MODEL from a local .env file

logging.basicConfig(
    filename="agent.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    agent = ReActAgent(verbose=True)
    print("ReAct Agent ready. Type 'exit' to quit.\n")
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Done.")
            break
        if not user_query:
            continue
        final_answer = agent.run(user_query)
        print(f"\nAgent: {final_answer}\n")


if __name__ == "__main__":
    main()
