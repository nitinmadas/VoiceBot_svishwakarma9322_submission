import pandas as pd
from modules.response_gen import agent
import os
import datetime
import time

# Make sure the output directory exists
os.makedirs("output", exist_ok=True)

def run_inference():
    # Ask user for path
    test_path = input("📂 Enter the path to your test CSV file: ").strip()

    # Load the test questions
    try:
        df = pd.read_csv(test_path)
        print(df.head())
    except FileNotFoundError:
        print("❌ Error: File not found.")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    if 'Questions' not in df.columns:
        print("❌ Error: 'Questions' column not found in the provided file.")
        return
    delay = 1  # seconds
    print(f"⏳ Each query will be delayed by {delay} second(s) to avoid rapid requests.")
    responses = []
    
    config = {"configurable": {"thread_id": "lendencol-thread-2"}}

    for question in df['Questions']:
        response = agent.invoke(    
             {"messages": [{"role": "user", "content": str(question)}]},
             config=config
                )  # Ensure it's string

        responses.append(response["messages"][-1].content)
        print(f"Question: {question}\nResponse: {response}\n")

    df['Responses'] = responses

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/predictions_{timestamp}.csv"
    time.sleep(delay)  # Add a 1 second delay to avoid rapid requests
    df.to_csv(output_path, index=False)
    print(f"✅ Inference complete. Output saved to {output_path}")

if __name__ == "__main__":
    run_inference()



