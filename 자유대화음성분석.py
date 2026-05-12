import json
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# --- 설정 (환경에 맞춰서 변경) ---
JSON_ROOT = r"D:\자유대화 음성(일반남녀)\Training\[라벨]1.AI챗봇"
WAV_ROOT = r"D:\자유대화 음성(일반남녀)\Training"
SAVE_PATH = r"D:\SpeakFit_Training_Master - 복사본.csv"

# 끊어갈 개수 설정 (기본 10만개)
BATCH_SIZE = 100000 

def extract_audio_features(wav_path, gender):
    try:
        # 가볍게 10초만 로드 (성능을 위해 sr=16000 고정)
        y, sr = librosa.load(wav_path, sr=16000, duration=10) 
        if len(y) == 0: return np.nan
        
        # 1. Pitch 추출 (YIN 알고리즘 사용)
        pitches = librosa.yin(y, fmin=65, fmax=450)
        valid_pitches = pitches[pitches > 0]
        p_median = np.median(valid_pitches) if len(valid_pitches) > 0 else np.nan
        
        # 성별에 따른 유효 피치 범위 필터링
        if gender == '남':
            if not (70 <= p_median <= 220): p_median = np.nan
        elif gender == '여':
            if not (140 <= p_median <= 380): p_median = np.nan
        
        return p_median # Pitch 값만 반환
    except:
        return np.nan

def run_speakfit_relay():
    # 1. 기존 작업 결과 확인 (이어하기 로직)
    final_results = []
    already_done_count = 0
    if os.path.exists(SAVE_PATH):
        try:
            old_df = pd.read_csv(SAVE_PATH, encoding='utf-8-sig')
            final_results = old_df.to_dict('records')
            already_done_count = len(final_results)
            print(f"🔄 이미 {already_done_count}건이 완료되었습니다. 다음부터 시작합니다.")
        except:
            print("⚠️ 기존 파일을 읽지 못해 새로 시작합니다.")

    # 2. 파일 목록 생성
    print("🔍 파일 목록 스캔 중...")
    all_json_paths = []
    for root, _, files in os.walk(JSON_ROOT):
        for f in files:
            if f.lower().endswith('.json'):
                all_json_paths.append(os.path.join(root, f))
    
    # 작업할 범위 선택
    todo_json_paths = all_json_paths[already_done_count : already_done_count + BATCH_SIZE]
    
    if not todo_json_paths:
        print("🎉 모든 데이터 분석이 완료되었습니다")
        return

    print(f"🚀 이번 실행 목표: {already_done_count + 1}번부터 {already_done_count + len(todo_json_paths)}번까지 분석")

    # 3. WAV 파일 위치 딕셔너리 생성 (속도 향상을 위해 메모리에 매핑)
    wav_dict = {}
    print("🔍 WAV 위치 확인 중...")
    for root, _, files in os.walk(WAV_ROOT):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_dict[f.rsplit('.', 1)[0]] = os.path.join(root, f)

    # 4. 분석 시작
    for i, j_path in enumerate(tqdm(todo_json_paths, desc="분석 진행 중")):
        try:
            with open(j_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            fname_key = data['발화정보']['fileNm'].rsplit('.', 1)[0]
            gender = data['녹음자정보']['gender']
            region = data['대화정보']['cityCode']
            stt_text = data['발화정보']['stt']
            duration = float(data['발화정보']['recrdTime'])
            
            wav_path = wav_dict.get(fname_key)
            if not wav_path: continue
            
            # 오디오에서 Pitch만 추출
            p_val = extract_audio_features(wav_path, gender)
            
            # 유효한 피치 데이터가 있을 때만 결과에 추가
            if not np.isnan(p_val):
                # 2. WPM 계산 (텍스트 길이와 발화 시간을 기반으로 계산)
                text_len = len(str(stt_text).replace(" ", ""))
                wpm_val = text_len / (duration / 60) if duration > 0 else 0
                
                region_nm = "표준어" if region == "수도권" else region
                
                final_results.append({
                    'GroupID': already_done_count + i + 1,
                    'match_key': fname_key,
                    'Group': f"{region_nm}_{gender}성",
                    'Pitch': p_val,
                    'WPM': wpm_val
                })

        except: continue

        # 5,000건마다 자동 저장 (안전장치)
        if (i + 1) % 5000 == 0:
            pd.DataFrame(final_results).to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')

    # 5. 최종 결과 저장
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')
    print(f"\n✅ 이번 분석 완료! 현재 총 {len(final_results)}건 저장됨.")
    print("💻 코드를 다시 실행하면 다음 10만 개를 이어서 분석합니다.")

if __name__ == "__main__":
    run_speakfit_relay()