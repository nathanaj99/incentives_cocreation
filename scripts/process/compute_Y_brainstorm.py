"""
Y_brainstorm.csv (Study 1 only): similarity of each participant's
pre-writing brainstorm to the other brainstorms in their group -- the
baseline for the difference-in-differences in Fig. S11.

The brainstorm is stored as JSON ({prompt: answer or list of answers}); it
is flattened to "prompt\\n- item\\n- item" sections before comparison.
Participants who left every brainstorm field empty are excluded (two in
Study 1; their brainstorm is stored as a Python-dict string of blanks).

Input:  raw/submissions.csv
Output: processed/Y_brainstorm.csv  (rows ordered AI-DIV, SELF-DIV, AI-CONV, SELF-CONV)
"""

from common import GROUP_ORDER, load_submissions, parse_json_cell, output, parse_args, should_skip
from similarity import add_metric_args, write_y, y_frame


def format_brainstorm(data):
    """Flatten the brainstorm JSON into plain text."""
    if isinstance(data, dict):
        sections = []
        for key, value in data.items():
            if isinstance(value, list):
                content = "\n".join(f"- {item}" for item in value)
            elif isinstance(value, dict):
                content = format_brainstorm(value)
            else:
                content = str(value)
            sections.append(f"{key}\n{content}")
        return "\n\n".join(sections)
    if isinstance(data, list):
        return "\n".join(f"- {item}" for item in data)
    return str(data)


def _nonempty(value):
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return bool(str(value).strip())


def main():
    args = parse_args(__doc__, studies=(1,), extra=add_metric_args)
    for study in args.study:
        print(f"Study {study}")
        out = output(study, "Y_brainstorm.csv")
        if should_skip(out, args.force):
            continue
        submissions = load_submissions(study)
        brainstorms = submissions["brainstorm_text"].apply(parse_json_cell)
        keep = brainstorms.apply(lambda b: isinstance(b, dict) and any(_nonempty(v) for v in b.values()))
        submissions = submissions[keep].copy()
        submissions["brainstorm_text"] = brainstorms[keep].apply(format_brainstorm)
        frame = y_frame(submissions, "brainstorm_text", GROUP_ORDER, args.metrics, args.device)
        write_y(frame, out, args.metrics)


if __name__ == "__main__":
    main()
