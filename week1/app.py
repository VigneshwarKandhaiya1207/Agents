import os
import requests
import json
import gradio as gr
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "user": os.getenv("PUSHOVER_USER"),
            "token": os.getenv("PUSHOVER_TOKEN"),
            "message": text
        }
    )

def record_user_details(email,name="Name not provided",notes="Notes not provided"):
    push(f"Recording the interest from user {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording question {question} that I could not answer")
    return {"recorded": "ok"}

record_user_details_json={
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being touch and provide an email address",
    "parameters":{
        "type": "object",
        "properties":{
            "email":{
                "type": "string",
                "description": "The email address of the user"
            },
            "name":{
                "type": "string",
                "description": "The name of the user"
            },
            "notes":{
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }        
        },
        "required":["email"],
        "additionalProperties": False
    }
}


record_unknown_question_json={
    "name": "record_unknown_question",
    "description": "Always use this tool to record the questions that you are unable to answer as you do not have the context",
    "parameters":{
        "type": "object",
        "properties":{
            "question":{
                "type": "string",
                "description": "The question that could not be answered"
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}


tools=[
    {"type": "function","function": record_user_details_json},
    {"type": "function","function": record_unknown_question_json}
]

class Me:

    def __init__(self):
        self.openai=OpenAI()
        self.name="Vigneshwar Kandhaiya"
        reader=PdfReader("vignesh_linkedin.pdf")
        self.linkedin=""
        for page in reader.pages:
            text=page.extract_text()
            if text:
                self.linkedin+=text

        with open("vigneshwar_kandhaiya_summary.txt", "r") as f:
            self.summary=f.read()
    
    def handle_tool_calls(self,tool_calls):
        results=[]
        for tool_call in tool_calls:
            tool_name=tool_call.function.name
            arguments=json.loads(tool_call.function.arguments)
            print(f"Tool Called : {tool_name}", flush=True)
            tool=globals().get(tool_name)
            result=tool(**arguments) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})

        return results
    
    def system_prompt(self):
        system_prompt=f""" You are acting as a {self.name} . You are answering questions on {self.name}'s profile.
Particularly, questions related to the {self.name}'s career, expectation,background and skills.
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible.
You are provided with a summary of {self.name}'s background and LinkedIn profile which you can use to answer the questions.
Be professional and engaging , as if you are talking to a potential client or a Future employer who wants to know about you.
If you don't know the answer to a question, use your record_unknown_question tool to record questions that you couldn't answer or even trivial or unrelated to career.
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

\n\n ## Summary : \n {self.summary} \n\n ## LinkedIn : \n {self.linkedin}
With this context , please chat with the user ,always staying in character as {self.name}
"""
        return system_prompt
    
    def chat(self,message,history):
        messages=[{"role": "system","content": self.system_prompt()}] + history + [{"role": "user","content":message}]
        done=False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            finish_reason=response.choices[0].finish_reason

            if finish_reason=="tool_calls":
                message=response.choices[0].message
                tool_calls=message.tool_calls
                results=self.handle_tool_calls(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done=True
        return response.choices[0].message.content
    

if __name__=="__main__":
    me=Me()
    gr.ChatInterface(me.chat,type='messages').launch()