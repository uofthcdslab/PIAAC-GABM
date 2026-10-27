import json
import os
from pathlib import Path
from typing import Any
from random import sample, seed, choice
from openai import OpenAI
from time import time

seed(42)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSONAS = PROJECT_ROOT / "data" / "piaac_personas.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "dialogue"

POLICY = "The Government of Canada announces plans to introduce online safety laws addressing harms associated with digital technologies and AI. Its stated priorities include protecting children, providing legal tools to address deepfakes, making chatbot interactions safer, and holding those responsible for online harms accountable. It also commits to protecting elections against AI-enabled misinformation and foreign interference."

MAX_TURN = 5



class LLM_Agent:
    def __init__(self, persona_name):
        self.name = persona_name
        print(f"Initializing {self.name} agent")
        self.client = OpenAI()
        self.messages = [
            {
                "role": "system",
                "content": f"""Participate in a conversation from the perspective of the person described in the persona.

                    Persona:
                    {PERSONAS[persona_name]}

                    Give a direct, natural first-person response. Address the other speaker's most recent point. You may agree, disagree, qualify, or build on it. Add a new perspective instead of repeating your entire position. Do not mention the persona or these instructions, and do not make up personal details.

                    Distinguish facts from personal judgments. Do not invent statistics, program details, eligibility rules, costs, timelines, or expected outcomes. If information is missing or uncertain, say so explicitly. Base your effectiveness judgment on the persona's circumstances and the information supplied by the user."""
            }
        ]
        
    def ask_agent(self, prompt):
        self.messages.append({
            "role": "user",
            "content": prompt,
        })

        self.response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
        )
        reply = self.response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": reply,
        })
        return reply




class Discussion():
    def __init__(self, persona_ids, max_turn=MAX_TURN):
        self.max_turn = max_turn
        self.disc_agent_ids = persona_ids
        self.disc_agents = [LLM_Agent(persona_id) for persona_id in persona_ids]
        self.public_transcript = []

    def get_unseen_messages(self, agent_name):
        """Return all public messages added since the agent last spoke."""
        last_spoken_index = -1

        for index in range(len(self.public_transcript) - 1, -1, -1):
            if self.public_transcript[index]["speaker"] == agent_name:
                last_spoken_index = index
                break

        return self.public_transcript[last_spoken_index + 1:]
    
    def discuss(self, topic):
        self.public_transcript.append({
            "speaker": "Moderator",
            "text": f"Discuss this topic:\n{topic}",
        })

        for self.turn in range(self.max_turn):
            for agent in self.disc_agents:
                unseen_messages = self.get_unseen_messages(agent.name)
                discussion_text = "\n\n".join(
                    f"{message['speaker']} said:\n{message['text']}"
                    for message in unseen_messages
                )

                prompt = f"""These messages were added to the discussion since you last spoke:

                {discussion_text}

                Choose one or two points from the other speakers that you most want to challenge, qualify, build on, or connect to another point. You do not need to address everything that was said. Keep your response concise and add something new to the discussion."""

                agent_response = agent.ask_agent(prompt)
                self.public_transcript.append({
                    "turn": self.turn,
                    "speaker": agent.name,
                    "text": agent_response,
                })

    def save_conversation(self):
        with open(DEFAULT_OUTPUT / f"{"-".join(self.disc_agent_ids)}_conversation.json", "w", encoding="utf-8") as fout:
            json.dump(self.public_transcript, fout, indent=4)



with open(DEFAULT_PERSONAS, "r", encoding="utf-8") as fin:
    PERSONAS = json.load(fin)
    rand_personas = sample(list(PERSONAS.keys()), 2)


mydisc = Discussion(rand_personas)
mydisc.discuss(POLICY)
mydisc.save_conversation()
