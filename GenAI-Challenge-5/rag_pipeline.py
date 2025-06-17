# rag_pipeline.py

import pandas as pd
from google.cloud import bigquery
from vertexai.preview.language_models import ChatModel
from vertexai.preview.generative_models import GenerativeModel, HarmCategory, SafetySetting
from vertexai.evaluation import (
    EvalTask,
    MetricPromptTemplateExamples,
)
import vertexai
import datetime

# --- Configuration ---
PROJECT_ID = "qwiklabs-gcp-02-4c9c7fb5e8ec"
LOCATION = "us-central1"
BQ_DATASET = "alaska_dept_of_snow"
TABLE_RAW = "faq_data"
TABLE_EMBEDDED = "faq_data_embedded"
EMBED_MODEL = "faq_embeddings"
TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET}.{TABLE_EMBEDDED}"
RAW_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET}.{TABLE_RAW}"
EMBED_MODEL_ID = f"{PROJECT_ID}.{BQ_DATASET}.{EMBED_MODEL}"

vertexai.init(project=PROJECT_ID, location=LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)

# --- Generative Models ---
checker_model = GenerativeModel("gemini-2.0-flash-001")
responder_model = GenerativeModel(
    "gemini-2.0-flash",
    safety_settings=[
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold="BLOCK_LOW_AND_ABOVE"),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold="BLOCK_LOW_AND_ABOVE"),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold="BLOCK_LOW_AND_ABOVE"),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold="BLOCK_LOW_AND_ABOVE"),
    ],
    system_instruction=[
        "You are a helpful and polite bot assisting citizens of Alaska with commonly asked questions. "
        "Greet the user warmly."
    ]
)
chat = responder_model.start_chat()

# --- Sensitive Input Check ---
def is_safe_input(input_text):
    prompt = f"""
    You are a sensitive information checker.

    Your job is to analyze the following input and determine whether it contains any sensitive information, such as:
    - Name, Phone number, Email, Address, Government ID, Credit card, Bank info

    If it contains sensitive information, return exactly: NO
    If it does NOT contain sensitive information, return exactly: YES

    Text to analyze:
    \"\"\"{input_text}\"\"\"
    """
    check = checker_model.generate_content(prompt).text.strip().upper()
    return "YES" if check == "YES" else "NO"

# --- Vector Search ---
def fetch_faq_results(user_question):
    query = f"""
    SELECT
      query.query,
      result.base.question,
      result.base.answer,
      result.distance
    FROM VECTOR_SEARCH(
      TABLE `{TABLE_ID}`,
      'embedding',
      (
        SELECT
          ml_generate_embedding_result AS embedding,
          '{user_question}' AS query
        FROM ML.GENERATE_EMBEDDING(
          MODEL `{EMBED_MODEL_ID}`,
          (SELECT '{user_question}' AS content)
        )
      ),
      top_k => 3,
      options => '{{"fraction_lists_to_search": 1.0}}'
    ) AS result
    """
    return bq_client.query(query).to_dataframe()

# --- Generate Chat Response ---
def generate_bot_response(user_input):
    results = fetch_faq_results(user_input)
    context = "\n\n".join([f"Q: {row['question']}\nA: {row['answer']}" for _, row in results.iterrows()])
    prompt = f"You are a helpful assistant for the citizen and residents of Alaska. Use the following FAQ context to answer:\n\n{context}\n\nUser: {user_input}"
    return responder_model.generate_content(prompt)

# --- Secure Chat Flow ---
def chat_secure():
    print("🤖 Hello! I'm your Alaska Help Bot. Ask me anything related to Alaska. Type 'exit' to end the session.\n")

    while True:
        prompt = input("You: ")
        if prompt.strip().lower() in ["exit", "quit"]:
            print("👋 Session ended. Stay safe and take care!")
            break

        if is_safe_input(prompt) != "YES":
            print("🚫 Rejected: Your question contains sensitive information.")
            continue

        try:
            response = generate_bot_response(prompt)

            if response.candidates and response.candidates[0].finish_reason == "SAFETY":
                print("⚠️ Gemini blocked this response due to safety policies.")
                continue

            response_text = response.text.strip()
            if is_safe_input(response_text) != "YES":
                print("🚫 Sorry. The response contains sensitive information and cannot be shown.")
                continue

            print("Gemini:", response_text)

        except Exception as e:
            print("❗ Error occurred:", str(e))

# --- Run Evaluation ---
def run_eval():
    example_dataset = [
        {
            "prompt": "How many people does ADS serve?",
            "answer": "ADS serves approximately 750,000 people across Alaska's widely distributed communities and remote areas."
        },
        {
            "prompt": "Does ADS use cloud services for its data?",
            "answer": "ADS is exploring cloud options for real-time data sharing, but some administrators have reservations about security, compliance, and ongoing costs."
        },
        {
            "prompt": "What kind of vehicles does ADS operate?",
            "answer": "ADS operates a fleet of snowplows, graders, and specialized 'snow blowers' designed for extreme weather. Some remote regions also use tracked vehicles."
        },
        {
            "prompt": "How do I volunteer to help with community snow events?",
            "answer": "Check your local ADS district’s website or bulletin board. Some regions host volunteer programs for sidewalk clearing and elderly assistance."
        },
        {
            "prompt": "How can I check current road conditions statewide?",
            "answer": "Use the ADS 'SnowLine' app or visit the official ADS website’s road conditions dashboard, which is updated hourly with closures and warnings."
        },
    ]

    eval_dataset = pd.DataFrame([
        {
            "instruction": "You are a helpful and polite bot assisting citizens of Alaska with commonly asked questions. Greet the user warmly.",
            "prompt": f"You are a helpful assistant for the citizen and residents of Alaska. Use the following FAQ context to answer: {item['prompt']}",
            "context": f"Answer: {item['prompt']}",
            "response": item["answer"],
        }
        for item in example_dataset
    ])

    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_task = EvalTask(
        dataset=eval_dataset,
        metrics=[
            MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
            MetricPromptTemplateExamples.Pointwise.VERBOSITY,
            MetricPromptTemplateExamples.Pointwise.INSTRUCTION_FOLLOWING,
            MetricPromptTemplateExamples.Pointwise.SAFETY,
        ],
        experiment=f"alaska-dept-of-snow-faqs-{run_ts}",
    )

    prompt_template = "Instruction: {instruction}. Prompt: {context}. Post: {response}"
    result = eval_task.evaluate(
        prompt_template=prompt_template,
        experiment_run_name=f"alaska-dept-of-snow-faqs-{run_ts}"
    )
    return result

# Entry Point
if __name__ == "__main__":
    chat_secure()
