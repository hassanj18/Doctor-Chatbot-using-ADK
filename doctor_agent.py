from google.adk import Agent # Import the Agent class
# from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk import Runner
from google.genai import types # For creating message Content/Parts
from typing import Dict, Any, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import asyncio
from pinecone_utility import search_kb
# Load environment variables
load_dotenv()

# Define constants
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"
AGENT_MODEL = MODEL_GEMINI_2_0_FLASH
APP_NAME = "doctor_assistant_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

async def schedule_appointment(patient_info: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule a medical appointment"""
    try:
        # Add appointment to memory
        appointments = []  # Global list to store appointments
        appointment_id = len(appointments) + 1
        appointment = {
            "id": appointment_id,
            "patient_name": patient_info.get("name"),
            "date": patient_info.get("date"),
            "time": patient_info.get("time"),
            "status": "scheduled"
        }
        appointments.append(appointment)
        return appointment
    except Exception as e:
        return {"error": str(e)}

async def send_confirmation_email(appointment_info: Dict[str, Any]) -> bool:
    """Send appointment confirmation email"""
    try:
        # Email configuration
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        receiver_email = appointment_info.get('email')
        
        # Create message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = 'Medical Appointment Confirmation'
        
        body = f"""Dear {appointment_info['patient_name']},

Your medical appointment has been successfully scheduled:
Date: {appointment_info['date']}
Time: {appointment_info['time']}

Please arrive 15 minutes before your scheduled time.

Best regards,
Medical Team
"""
        
        message.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

async def Search_KB(query: str, top_k: int = 5) -> list:
    """Search the knowledge base for relevant Q&A using semantic search."""
    return search_kb(query, top_k=top_k)

# Initialize agent
doctor_agent = Agent(
    name="doctor_agent_v1",
    model=AGENT_MODEL,
    description="Provides medical information and appointment scheduling services.",
    instruction="You are an AI assistant for an ENT (Ear, Nose, Throat) specialist. Your job is to reduce unnecessary patient calls by helping users through chat. You must follow these rules:\n\nAnswer Using the Knowledge Base:\nIf the user's question is related to common or mild ENT issues — such as sinus pain, cold, minor ear discomfort, allergy, sore throat, blocked nose, or similar non-critical symptoms — respond clearly using the existing knowledge base. Use the Search_KB tool to look up relevant information.\n\nDo Not Provide Medical Advice Beyond ENT:\nIf the question is unrelated to ENT or requires expertise outside this domain, politely inform the user that this assistant is for ENT-specific concerns only.\n\nEscalate to Appointment Booking ONLY IF:\n\nThe symptoms described are severe, persistent, or worsening, such as:\n- Difficulty breathing\n- Hearing loss\n- High fever with pain\n- Blood discharge\n- Vertigo or extreme dizziness\nThe question is urgent, unclear, or seems beyond general advice\nThe user explicitly requests to book an appointment\n\nAppointment Booking Workflow:\nIf escalation is needed, collect the user's:\n- Full name\n- Age\n- email\n- Symptom summary\n- Preferred time or availability\nThen pass the booking details to the appointment system or mark for follow-up. also send a booking confirmation email( this is the email provided by the user ) to the user using gmail tool. and mark his google calender on that time and date.\n\nAlways stay respectful, concise, and clear.",
    tools=[
        schedule_appointment,
        send_confirmation_email,
        Search_KB
    ]
)

# Initialize session service
session_service = InMemorySessionService()

# Create session
session = asyncio.run(session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
))

# Initialize runner
runner = Runner(
    agent=doctor_agent,
    app_name=APP_NAME,
    session_service=session_service
)

async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response." # Default

    # Run the agent and process events
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    print(f"<<< Agent Response: {final_response_text}")

async def run_conversation():
    """Run a sample conversation with the agent."""
    await call_agent_async("I keep coughing at night. Could it be related to my throat", runner, USER_ID, SESSION_ID)

if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")
