import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# -----------------------
# Helper functions
# -----------------------
def add_missing(series, frac=0.05, rng=None):
    """Inject NaN values randomly into a pandas Series."""
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(len(series)) < frac
    s = series.copy()
    s[mask] = np.nan
    return s

def choose_with_probs(choices, probs, size, rng):
    return rng.choice(choices, size=size, p=probs)

# -----------------------
# Dataset generator
# -----------------------
def generate_dataset(n_rows=100_000, seed=42):
    rng = np.random.default_rng(seed)

    # Pools and probabilities
    genders = ["Male", "Female", "Other"]
    gender_probs = [0.55, 0.42, 0.03]

    education_levels = ["High School", "Associate Degree", "Bachelor's", "Master's", "PhD"]
    education_probs = [0.25, 0.15, 0.35, 0.20, 0.05]

    employment_statuses = ["Employed", "Unemployed", "Student", "Retired", "Self-Employed"]
    employment_probs = [0.60, 0.10, 0.15, 0.10, 0.05]

    risk_tolerances = ["Low", "Medium", "High"]
    risk_probs = [0.35, 0.45, 0.20]

    races = ["White", "Black or African American", "Asian", "Hispanic or Latino", "Other"]
    race_probs = [0.55, 0.12, 0.15, 0.15, 0.03]

    city_tiers = ["Tier 1", "Tier 2", "Tier 3"]
    city_probs = [0.20, 0.40, 0.40]

    investment_options = [
        "Mutual Funds", "Equity Market", "Debentures", "Government Bonds",
        "Fixed Deposits", "Public Provident Fund", "Gold"
    ]
    rankings = np.array([1, 2, 3, 4, 5, 6, 7])

    holding_periods = ["Less than 1 year", "1-3 years", "3-5 years", "More than 5 years"]
    holding_probs = [0.15, 0.35, 0.30, 0.20]

    monitoring_frequencies = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    monitoring_probs = [0.10, 0.30, 0.40, 0.15, 0.05]

    return_expectations = ["Less than 10%", "10%-20%", "20%-30%", "More than 30%"]
    return_probs = [0.30, 0.40, 0.20, 0.10]

    savings_objectives = ["Retirement Plan", "Health Care", "Children's Education", "Wealth Accumulation"]
    savings_probs = [0.35, 0.20, 0.15, 0.30]

    reasons_equity = ["Capital Appreciation", "Dividend"]
    reasons_equity_probs = [0.8, 0.2]

    reasons_mutual = ["Better Returns", "Tax Benefits", "Fund Diversification"]
    reasons_mutual_probs = [0.5, 0.2, 0.3]

    reasons_bonds = ["Safe Investment", "Assured Returns", "Tax Incentives"]
    reasons_bonds_probs = [0.5, 0.3, 0.2]

    reasons_fd = ["Fixed Returns", "High Interest Rates", "Risk Free"]
    reasons_fd_probs = [0.5, 0.25, 0.25]

    sources_info = ["Newspapers and Magazines", "Financial Consultants", "Television", "Internet"]
    sources_probs = [0.10, 0.25, 0.15, 0.50]

    invest_in_avenues = ["Yes", "No"]
    invest_in_avenues_probs = [0.80, 0.20]

    # Risk-conditioned "main investment avenue"
    base_main_probs = np.array([0.35, 0.25, 0.05, 0.08, 0.12, 0.05, 0.10])
    risk_to_probs = {
        "Low":   np.array([0.30, 0.08, 0.08, 0.15, 0.18, 0.12, 0.09]),
        "Medium":np.array([0.36, 0.20, 0.06, 0.10, 0.14, 0.06, 0.08]),
        "High":  np.array([0.28, 0.40, 0.04, 0.05, 0.08, 0.05, 0.10]),
    }

    # -----------------------
    # Core respondent fields
    # -----------------------
    df = pd.DataFrame({
        "GENDER": choose_with_probs(genders, gender_probs, n_rows, rng),
        "AGE": rng.integers(18, 66, n_rows),
        "Do you invest in Investment Avenues?": choose_with_probs(invest_in_avenues, invest_in_avenues_probs, n_rows, rng),
    })

    # Rankings per respondent
    rank_matrix = np.vstack([rng.permutation(rankings) for _ in range(n_rows)])
    for j, option in enumerate(investment_options):
        col = f"What do you think are the best options for investing your money? (Rank in order of preference) [{option}]"
        df[col] = rank_matrix[:, j]

    # Other fields
    df["How long do you prefer to keep your money in any investment instrument?"] = choose_with_probs(holding_periods, holding_probs, n_rows, rng)
    df["How often do you monitor your investment?"] = choose_with_probs(monitoring_frequencies, monitoring_probs, n_rows, rng)
    df["How much return do you expect from any investment instrument?"] = choose_with_probs(return_expectations, return_probs, n_rows, rng)
    df["What are your savings objectives?"] = choose_with_probs(savings_objectives, savings_probs, n_rows, rng)
    df["Reasons for investing in Equity Market"] = choose_with_probs(reasons_equity, reasons_equity_probs, n_rows, rng)
    df["Reasons for investing in Mutual Funds"] = choose_with_probs(reasons_mutual, reasons_mutual_probs, n_rows, rng)
    df["Reasons for investing in Government Bonds"] = choose_with_probs(reasons_bonds, reasons_bonds_probs, n_rows, rng)
    df["Reasons for investing in Fixed Deposits"] = choose_with_probs(reasons_fd, reasons_fd_probs, n_rows, rng)
    df["Your sources of information for investments is"] = choose_with_probs(sources_info, sources_probs, n_rows, rng)

    # Demographics
    df["Education Level"] = choose_with_probs(education_levels, education_probs, n_rows, rng)
    df["Annual Income"] = rng.normal(60000, 15000, n_rows).round(2).clip(10_000, 200_000)
    df["Employment Status"] = choose_with_probs(employment_statuses, employment_probs, n_rows, rng)
    df["Risk Tolerance"] = choose_with_probs(risk_tolerances, risk_probs, n_rows, rng)
    df["Race"] = choose_with_probs(races, race_probs, n_rows, rng)
    df["City Tier"] = choose_with_probs(city_tiers, city_probs, n_rows, rng)

    # Main avenue conditioned on risk
    risk_vals = df["Risk Tolerance"].to_numpy()
    main_choice = []
    for r in risk_vals:
        probs = risk_to_probs.get(r, base_main_probs)
        main_choice.append(rng.choice(investment_options, p=probs))
    df["Which investment avenue do you mostly invest in?"] = main_choice

    # -----------------------
    # Comments (vectorized style)
    # -----------------------
    aggressive = np.array([
        "I'm excited for high returns, even with higher risks.",
        "Willing to take risks for better profits.",
        "Aggressive growth is my goal!"
    ])
    conservative = np.array([
        "I prefer safety over high returns.",
        "Conservative investments make me feel secure.",
        "Stability is more important to me."
    ])
    moderate = np.array([
        "I'm open to some risk, but not too much.",
        "Looking for a balance between risk and return.",
        "Moderate returns with controlled risk suit me."
    ])
    mixed = np.array([
        "Just starting to invest, still learning.",
        "Happy with my investment so far.",
        "I want to know more about my options.",
        "Not sure if my portfolio is right.",
        "Wish I had more guidance on investments.",
        "I feel uncertain about the market trends.",
        "Looking for better investment strategies.",
        "Hope my returns improve next year.",
        "Need more advice for my financial planning.",
        "Sometimes I feel overwhelmed by choices.",
        "Trying to build my confidence as an investor.",
        "Satisfied but always looking for improvements.",
        "Would like more transparency about risks.",
        "Concerned about recent market volatility.",
        "Investing feels complicated at times.",
        "Feeling optimistic about the future.",
        "Unsure if my asset allocation fits my goals.",
        "Curious about new investment opportunities.",
        "Wish I had started investing earlier.",
        "I'm cautious but interested in learning more."
    ])

    ret = df["How much return do you expect from any investment instrument?"].to_numpy()
    risk = df["Risk Tolerance"].to_numpy()

    mask_aggr = (risk == "High") & np.isin(ret, ["20%-30%", "More than 30%"])
    mask_cons = (risk == "Low") & (ret == "Less than 10%")
    mask_modr = (risk == "Medium")

    comments = rng.choice(mixed, size=n_rows)
    comments[mask_aggr] = rng.choice(aggressive, size=mask_aggr.sum())
    comments[mask_cons] = rng.choice(conservative, size=mask_cons.sum())
    comments[mask_modr] = rng.choice(moderate, size=mask_modr.sum())
    # inject 1.5% missing
    drop_mask = rng.random(n_rows) < 0.015
    comments = pd.Series(comments)
    comments[drop_mask] = np.nan
    df["Comment"] = comments

    # -----------------------
    # Missing values
    # -----------------------
    missing_plan = {
        "GENDER": 0.02,
        "AGE": 0.01,
        "Do you invest in Investment Avenues?": 0.03,
        "How long do you prefer to keep your money in any investment instrument?": 0.04,
        "How often do you monitor your investment?": 0.03,
        "How much return do you expect from any investment instrument?": 0.03,
        "Which investment avenue do you mostly invest in?": 0.04,
        "What are your savings objectives?": 0.04,
        "Reasons for investing in Equity Market": 0.02,
        "Reasons for investing in Mutual Funds": 0.02,
        "Reasons for investing in Government Bonds": 0.02,
        "Reasons for investing in Fixed Deposits": 0.02,
        "Your sources of information for investments is": 0.03,
        "Education Level": 0.05,
        "Annual Income": 0.05,
        "Employment Status": 0.05,
        "Risk Tolerance": 0.05,
        "Race": 0.03,
        "City Tier": 0.03,
    }
    for col, frac in missing_plan.items():
        df[col] = add_missing(df[col], frac=frac, rng=rng)

    for option in investment_options:
        col = f"What do you think are the best options for investing your money? (Rank in order of preference) [{option}]"
        df[col] = add_missing(df[col], frac=0.02, rng=rng)

    return df

# -----------------------
# Main CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100000, help="Number of rows")
    parser.add_argument("--out", type=str, default="Combined_Investment_Data.csv", help="Output CSV file path")
    args = parser.parse_args()

    df = generate_dataset(args.rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {args.rows:,} rows to {out_path}")
