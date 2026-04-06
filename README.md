# Two-Point Discrimination

A Streamlit-based examiner-assist app for **two-point discrimination testing**.

## Live App

Use the public app here:

**https://dreamycat925-2pd-2pd-discrimination-streamlit-app-2oanve.streamlit.app/**

## Overview

- Three phases: **Practice**, **Main test**, and **Post-test check**
- The main test uses a **100-trial** schedule with **40 one-point trials** and **60 two-point trials**
- No stimulus type repeats **4 or more times in a row**
- The staircase is updated **only on two-point trials**
- Start level: **30 mm**
- Fixed ladder: **1, 3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 35, 40, 45, 50 mm**
- The staircase moves to the **adjacent ladder level**
- Lower bound: **1 mm**
- Upper bound: **50 mm**
- Formal threshold: **median of the last 6 reversals**

## Practice / Post-test Check

- Starts at **30 mm**
- Uses **two-point trials only**
- Ends with **PASS** after **5 consecutive correct responses**
- If the participant makes **2 errors at the same mm**, the app steps up to the next higher level
- If the participant makes **2 errors at 50 mm**, the phase ends as **FAIL**

In usual use, the participant should achieve **5 consecutive correct responses in practice** before proceeding to the main test.

## Main Test

- Uses **2-down 1-up**
- Uses the fixed ladder above
- The app can use:
  - **Series 1**
  - **Series 2**
  - **Random**

### Main test stopping rules

- **PASS**: **4 consecutive correct responses** on **two-point trials at 1 mm**
- **FAIL**: **2 consecutive incorrect responses** on **two-point trials at 50 mm**
- **Converged**: **10 reversals**
- **Non-convergent**: **100 trials**

## Output

The app provides:

- A line chart of the main-test course
- A downloadable **TXT summary**
- A downloadable **CSV log**

## Recommended Workflow

1. Run **Practice**
2. Confirm **5 consecutive correct responses**
3. Run the **Main test**
4. Run the **Post-test check** if needed

## Local Run

```bash
pip install -r requirements.txt
streamlit run 2pd_discrimination_streamlit_app.py
```

## Notes

- This is an **examiner support tool**
- It is **not a medical device**
