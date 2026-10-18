"""
ai_requests.csv / ai_requests_explode.csv: the AI chat transcripts of the
AI-assisted participants.

  ai_requests.csv          one row per participant; `ai_requests` is the list
                           of their messages to the AI
  ai_requests_explode.csv  one row per request, with the AI's reply in
                           `ai_outputs` (input to label_ai_requests.py and
                           extract_drafts.py)

Input:  raw/submissions.csv
Output: processed/ai_requests.csv, processed/ai_requests_explode.csv
"""

from common import ai_submissions, output, parse_args, save_csv, split_transcript


def main():
    args = parse_args(__doc__)
    for study in args.study:
        print(f"Study {study}")
        ai = ai_submissions(study)
        split = ai["ai_transcript"].apply(split_transcript)
        ai["ai_requests"] = split.str[0]
        # Pad replies so every request has one (a trailing request may have
        # gone unanswered, e.g. two Study 1 transcripts)
        ai["ai_outputs"] = [outs + [None] * (len(reqs) - len(outs))
                            for reqs, outs in zip(split.str[0], split.str[1])]

        save_csv(ai[["participant_id", "ai_requests"]], output(study, "ai_requests.csv"))

        exploded = ai[["participant_id", "ai_requests", "ai_outputs"]].explode(["ai_requests", "ai_outputs"])
        save_csv(exploded, output(study, "ai_requests_explode.csv"))


if __name__ == "__main__":
    main()
