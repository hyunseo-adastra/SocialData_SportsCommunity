import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import csv
import os
import re

# --- 설정 부분 ---
GALLERY_ID = 'formula1'
START_PAGE = 10981
# 수집할 시간 범위 설정 (3월 3일 00:00 ~ 02:00)
SEARCH_START_TIME = datetime(2024, 6, 30, 22, 0, 0)
SEARCH_END_TIME = datetime(2024, 7, 1, 0, 0, 0)
# 크롤링을 중단할 날짜
STOP_DATE = datetime(2024, 7, 2)

OUTPUT_DIR = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData'
# 수집 내용에 맞춰 파일 이름 변경
FILENAME = 'Community_Austria.csv'
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, FILENAME)


# ---------------------------

def crawl_and_save_posts():
    """
    지정된 기간의 게시물 제목과 본문을 수집하여 CSV 파일로 저장합니다.
    (Selenium 없이 requests만 사용하여 안정성을 높였습니다.)
    """
    page = START_PAGE
    crawling = True
    results = []

    # 모든 요청을 관리할 Session 객체 생성
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    print(f"[{GALLERY_ID} 갤러리] - 제목 및 본문 수집 모드")
    print(f"수집 대상 시간: {SEARCH_START_TIME.strftime('%Y-%m-%d %H:%M')} ~ {SEARCH_END_TIME.strftime('%Y-%m-%d %H:%M')}")
    print(f"저장 위치: {OUTPUT_FILENAME}\n")

    while crawling and page > 0:
        url = f"https://gall.dcinside.com/mgallery/board/lists/?id={GALLERY_ID}&page={page}"
        try:
            # 1. 게시물 목록 페이지에 접속
            response = session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            post_list = soup.select('tr.ub-content.us-post')

            if not post_list:
                print(f"{page}페이지에 게시물이 없습니다. 크롤링을 중단합니다.")
                break

            for post in reversed(post_list):
                if 'notice' in post.select_one('.gall_num').text:
                    continue

                date_element = post.select_one('.gall_date')
                post_date = datetime.strptime(date_element.get('title'), '%Y-%m-%d %H:%M:%S')

                # 중단 날짜에 도달하면 크롤링 종료
                if post_date.date() >= STOP_DATE.date():
                    crawling = False
                    break

                # 목표 시간 범위에 해당하는 게시물인지 확인
                if SEARCH_START_TIME <= post_date < SEARCH_END_TIME:
                    title_element = post.select_one('.gall_tit a')
                    post_title = title_element.text.strip()
                    post_link = "https://gall.dcinside.com" + title_element['href']

                    print(f"\n발견: [{post_date.strftime('%Y-%m-%d %H:%M')}] {post_title}")

                    try:
                        # 2. 해당 게시물 페이지에 직접 접속하여 본문 수집
                        print("  > 본문 수집 중...")
                        post_page_res = session.get(post_link)
                        post_page_res.raise_for_status()
                        post_soup = BeautifulSoup(post_page_res.text, 'html.parser')

                        content_element = post_soup.select_one('div.writing_view_box')
                        post_content = content_element.get_text(separator='\n', strip=True) if content_element else ''

                        # 수집된 데이터를 results 리스트에 추가
                        results.append({
                            'post_timestamp': post_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'post_title': post_title,
                            'post_content': post_content
                        })
                        print("  > 수집 완료.")

                    except requests.RequestException as e:
                        print(f"  > [오류] 이 게시물의 본문을 가져오는 데 실패했습니다: {e}")
                        continue  # 실패 시 다음 게시물로 넘어감

            if not crawling:
                print(f"\n중단 날짜({STOP_DATE.strftime('%Y-%m-%d')})의 게시물에 도달하여 크롤링을 종료합니다.")
                break

            print(f"--- {page}페이지 스캔 완료 ---")
            page -= 1
            time.sleep(1)  # 서버 부하 방지를 위한 1초 대기

        except requests.RequestException as e:
            print(f"목록 페이지 접근 중 오류 발생: {e}")
            break

    # --- 수집된 모든 결과를 CSV 파일로 저장 ---
    if results:
        print(f"\n총 {len(results)}개의 게시물을 찾았습니다. 파일로 저장합니다.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILENAME, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # CSV 헤더(열 이름)를 수집 내용에 맞게 변경
            fieldnames = ['post_timestamp', 'post_title', 'post_content']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"저장 완료! '{os.path.abspath(OUTPUT_FILENAME)}' 에서 파일을 확인하세요.")
    else:
        print("\n수집된 게시물이 없어 파일을 저장하지 않았습니다.")


# --- 메인 실행 ---
if __name__ == "__main__":
    crawl_and_save_posts()
