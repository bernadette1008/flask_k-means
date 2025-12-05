from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import koreanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import base64
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 및 데이터 미리보기"""
    try:
        file = request.files['file']
        filename = file.filename.lower()

        # 파일 타입에 따라 읽기
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            # CSV 인코딩 자동 감지
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='cp949')
                except:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')

        # 데이터 미리보기 (최대 10행)
        head = df.head(10).fillna('-').to_dict(orient='records')
        columns = list(df.columns)

        # 숫자형 컬럼 및 변환 가능한 컬럼 찾기
        numeric_cols = []
        df_transformed = df.copy()
        col_map = {}

        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
                col_map[col] = col
            else:
                # 날짜형 변환 시도
                try:
                    tmp = pd.to_datetime(df[col], errors='coerce')
                    if tmp.notna().sum() > len(df) * 0.5:
                        transformed_col = f"{col}_timestamp"
                        df_transformed[transformed_col] = tmp.astype('int64') // 10**9
                        numeric_cols.append(transformed_col)
                        col_map[transformed_col] = col
                        continue
                except:
                    pass
                
                # 범주형 변환 시도
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:
                        le = LabelEncoder()
                        transformed_col = f"{col}_encoded"
                        df_transformed[transformed_col] = le.fit_transform(df[col].astype(str))
                        numeric_cols.append(transformed_col)
                        col_map[transformed_col] = col
                except:
                    continue

        return jsonify({
            "head": head,
            "columns": columns,
            "numeric_cols": numeric_cols,
            "col_map": col_map,
            "total_rows": len(df)
        })

    except Exception as e:
        return jsonify({"error": f"파일 읽기 오류: {str(e)}"}), 400


@app.route('/kmeans', methods=['POST'])
def kmeans_api():
    """K-Means 클러스터링 수행"""
    try:
        file = request.files['file']
        x_col = request.form['x_col']
        y_col = request.form['y_col']
        k = int(request.form['k'])

        if k < 2 or k > 20:
            return jsonify({"error": "k값은 2~20 사이여야 합니다."}), 400

        filename = file.filename.lower()
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='cp949')
                except:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')

        # 원본 데이터 저장 (클러스터 상세 정보용)
        df_original = df.copy()
        
        df_transformed = df.copy()
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    tmp = pd.to_datetime(df[col], errors='coerce')
                    if tmp.notna().sum() > len(df) * 0.5:
                        df_transformed[f"{col}_timestamp"] = tmp.astype('int64') // 10**9
                        continue
                except:
                    pass
                
                try:
                    le = LabelEncoder()
                    df_transformed[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                except:
                    continue

        if x_col not in df_transformed.columns:
            return jsonify({"error": f"X축 컬럼 '{x_col}'을 찾을 수 없습니다."}), 400
        if y_col not in df_transformed.columns:
            return jsonify({"error": f"Y축 컬럼 '{y_col}'을 찾을 수 없습니다."}), 400

        cols = [x_col, y_col]
        df_selected = df_transformed[cols].apply(pd.to_numeric, errors='coerce')
        
        # 유효한 인덱스 추적
        valid_indices = df_selected.dropna().index
        df_selected = df_selected.loc[valid_indices]

        if df_selected.shape[0] < k:
            return jsonify({"error": f"유효한 데이터가 {df_selected.shape[0]}개로 k값({k})보다 적습니다."}), 400

        if df_selected.shape[0] < 10:
            return jsonify({"error": "분석을 위한 데이터가 부족합니다 (최소 10개 필요)."}), 400

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(df_selected)

        stats = df_selected.copy()
        stats['cluster'] = labels
        
        cluster_stats = {}
        cluster_details = {}
        
        for cluster_id in range(k):
            cluster_mask = stats['cluster'] == cluster_id
            cluster_data = stats[cluster_mask]
            cluster_stats[str(cluster_id)] = {
                f"{x_col}_count": len(cluster_data),
                f"{x_col}_mean": float(cluster_data[x_col].mean()),
                f"{y_col}_mean": float(cluster_data[y_col].mean()),
            }
            
            # 클러스터에 속한 원본 데이터의 인덱스 가져오기
            cluster_indices = cluster_data.index
            
            # 원본 데이터에서 해당 인덱스의 모든 컬럼 데이터 가져오기
            original_cluster_data = df_original.loc[cluster_indices].copy()
            
            # NaN을 '-'로 변환하고 최대 1000개로 제한
            cluster_records = original_cluster_data.head(1000).fillna('-').to_dict(orient='records')
            cluster_details[str(cluster_id)] = cluster_records

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i in range(k):
            cluster_points = df_selected[labels == i]
            ax.scatter(cluster_points[x_col], cluster_points[y_col], 
                      c=colors[i % len(colors)], label=f'Cluster {i}', 
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        centers = model.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=300, 
                  alpha=0.8, marker='X', edgecolors='white', linewidth=2,
                  label='중심점')
        
        ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means 클러스터링 결과 (k={k})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({
            "image": img_base64,
            "cluster_stats": cluster_stats,
            "cluster_details": cluster_details,
            "total_points": int(df_selected.shape[0]),
            "removed_points": int(len(df) - df_selected.shape[0])
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"분석 중 오류 발생: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)