
from pathlib import Path
from email import policy
from email.parser import BytesParser

import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------------
# SPAMSHIELD AI - DATASET PREPARATION
# ----------------------------------

# Dataset folders are expected inside the project folder.
PROJECT_DIR = Path(__file__).parent

SPAM_DIR = PROJECT_DIR / "spam_2"
EASY_HAM_DIR = PROJECT_DIR / "easy_ham"
HARD_HAM_DIR = PROJECT_DIR / "hard_ham"
HAM_DIRS = [EASY_HAM_DIR, HARD_HAM_DIR]


def extract_email_text(file_path):
    """Extract readable plain-text content from an email file."""
    try:
        with open(file_path, "rb") as file:
            message = BytesParser(
                policy=policy.default
            ).parse(file)

        parts = []

        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        content = part.get_content()
                        if isinstance(content, str):
                            parts.append(content)
                    except Exception:
                        continue
        else:
            if message.get_content_type() == "text/plain":
                try:
                    content = message.get_content()
                    if isinstance(content, str):
                        parts.append(content)
                except Exception:
                    pass

        return "\n".join(parts).strip()

    except Exception as error:
        print(f"Skipping unreadable file {file_path.name}: {error}")
        return ""


def load_emails(folder, label):
    """Load email files from a folder and assign labels."""
    records = []

    if not folder.is_dir():
        return records

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        text = extract_email_text(file_path)

        if text:
            records.append({
                "text": text,
                "label": label
            })

    return records


def main():
    # Check whether all dataset folders exist.
    required_folders = [SPAM_DIR] + HAM_DIRS
    missing_folders = [
        folder for folder in required_folders
        if not folder.is_dir()
    ]

    if missing_folders:
        print("\nDataset folders not found!")
        print("Project folder:", PROJECT_DIR)
        print("\nMissing folders:")

        for folder in missing_folders:
            print(" -", folder)

        print(
            "\nMake sure spam2, easy ham, and hard ham "
            "are inside your project folder."
        )
        return

    print("Loading spam emails...")
    spam_records = load_emails(SPAM_DIR, "spam")

    print("Loading legitimate emails...")
    ham_records = []

    for folder in HAM_DIRS:
        ham_records.extend(load_emails(folder, "ham"))

    records = spam_records + ham_records

    if not records:
        print("No readable emails found. Check your dataset files.")
        return

    df = pd.DataFrame(records)

    # Remove empty messages and exact duplicate text.
    df = df[df["text"].str.strip().ne("")]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("\nSpamShield AI - Dataset Summary")
    print("--------------------------------")
    print(f"Spam emails: {(df['label'] == 'spam').sum()}")
    print(f"Legitimate emails: {(df['label'] == 'ham').sum()}")
    print(f"Total unique emails: {len(df)}")

    # Ensure both classes have enough examples.
    label_counts = df["label"].value_counts()

    if len(label_counts) < 2 or label_counts.min() < 2:
        print("\nNot enough emails from both classes to split the dataset.")
        return

    # Split into training and testing datasets.
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # Save CSV files in the project folder.
    train_df.to_csv(PROJECT_DIR / "emails_train.csv", index=False)
    test_df.to_csv(PROJECT_DIR / "emails_test.csv", index=False)

    print(f"\nTraining emails: {len(train_df)}")
    print(f"Testing emails: {len(test_df)}")
    print("\nCreated emails_train.csv and emails_test.csv")
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()