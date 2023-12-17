import openai


def isbookPass(userquery: str, bookinfo) -> bool:
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Based on the user's question about {user's question about the desired type of book} "
                        "and the provided information about the recommended book {recommended book information}, provide an evaluation of the recommendation. "
                        "Begin by explaining the alignment between the user's request and the recommended book, providing reasons to support your evaluation. "
                        "Then, conclude with your evaluation in the format 'Evaluation : P' (Positive) or 'Evaluation : F' (Negative). "
                        "If the evaluation is unclear or if the recommended book does not directly address the user's specific request, default to 'Evaluation : F'. "
                        "Please ensure that no sentences follow the evaluation result."
                    ),
                },
                {
                    "role": "user",
                    "content": f"user question:{userquery} recommendations:{bookinfo}",
                },
            ],
        )
    except openai.error.APIError as e:
        pf = "F"
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        pf = "F"
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.error.RateLimitError as e:
        pf = "F"
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass
    except:
        pf = "F"
        print("Unknown error while evaluating")
        pass

    pf = str(completion["choices"][0]["message"]["content"])
    ck = False
    for c in reversed(pf):
        if c == "P":
            return True
        elif c == "F":
            return False
    if ck == False:
        print("\nsmth went wrong\n")
        return False
