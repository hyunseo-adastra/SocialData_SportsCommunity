import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import logit
import os

# =========================================================
# 1. 설정 및 데이터 로드
# =========================================================
base_dir = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'
input_path = os.path.join(base_dir, 'AnalysisResults/f1_comment_level_dataset.xlsx')
output_txt_path = os.path.join(base_dir, 'AnalysisResults/f1_comment_logit_report.txt')

print(f"데이터 로드 중: {input_path}")
df = pd.read_excel(input_path)

df.columns = df.columns.str.strip()
print(f"총 댓글 수: {len(df)}개")
print(f"컬럼 목록: {list(df.columns)}")

# =========================================================
# 2. 로지스틱 회귀 함수 정의
# =========================================================
results_summary = []

def run_logit(target_col, title):
    print(f"\n--- Analyzing: {target_col} ({title}) ---")

    if target_col not in df.columns:
        print(f"[Skip] 컬럼 '{target_col}'이 데이터에 없습니다.")
        return

    # 종속변수는 반드시 0/1
    if not set(df[target_col].dropna().unique()).issubset({0, 1}):
        print(f"[Skip] '{target_col}'이 이진 변수가 아닙니다.")
        return

    formula = f"""
        {target_col} ~
        ev_unexp + ev_resp + ev_out +
        comment_length
    """

    try:
        model = logit(formula, data=df).fit(disp=False)

        summary_text = f"\n{'=' * 60}\n[Logit Model] {title}\nFormula: {formula}\n{'=' * 60}\n"
        summary_text += model.summary().as_text() + "\n\n"
        results_summary.append(summary_text)

        print(f"Pseudo R² (McFadden): {model.prsquared:.4f}")

        for term in ['ev_unexp', 'ev_resp', 'ev_out']:
            coef = model.params.get(term, 0)
            pval = model.pvalues.get(term, 1)
            odds = pd.np.exp(coef)
            star = "*" if pval < 0.05 else ""
            print(f" - {term}: Coef={coef:.4f}, OR={odds:.3f}, P={pval:.4f} {star}")

    except Exception as e:
        print(f"로지스틱 회귀 오류: {e}")