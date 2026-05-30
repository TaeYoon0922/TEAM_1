"""
jobkorea_crawler_fixed.py (Ultra-Fast Multithreaded Version)

설명:
- 잡코리아 합격자소서 상세정보를 크롤링하는 고성능 병렬 스크립트입니다.
- 1단계: 메인 브라우저에서 사용자가 로그인하면 그 쿠키(Session)를 추출합니다.
- 2단계: 이미지 로딩이 차단된 초경량 백업 headless 브라우저 4대를 동시에 구동합니다.
- 3단계: 추출한 로그인 세션 쿠키를 4대의 브라우저에 주입하여 동시에 4배 빠른 병렬 크롤링을 수행합니다.
- 4단계: 스레드 세이프(Thread-safe)하게 실시간으로 CSV/JSONL 파일에 추가(Append) 저장합니다.
"""

import argparse
import csv
import json
import random
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
    ElementClickInterceptedException,
)
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

STARTER_URL = "https://www.jobkorea.co.kr/starter/passassay?schTxt=&Page={page}"

# 파일 입출력 스레드 락
file_lock = threading.Lock()


def make_driver(headless: bool = False) -> Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    # 이미지 로딩 비활성화로 네트워크 대역폭 및 로딩 시간 3배 이상 절약
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    if headless:
        options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(15)
    return driver


def sleep_polite(min_sec: float = 1.0, max_sec: float = 2.0) -> None:
    time.sleep(random.uniform(min_sec, max_sec))


def text_or_empty(element: Optional[WebElement]) -> str:
    if element is None:
        return ""
    try:
        return element.text.strip()
    except Exception:
        return ""


def find_one(parent, by: By, value: str, default=None):
    try:
        return parent.find_element(by, value)
    except NoSuchElementException:
        return default


def close_possible_popups(driver: Chrome) -> None:
    candidates = [
        (By.ID, "closeIncompleteResume"),
        (By.CSS_SELECTOR, "button.close"),
        (By.CSS_SELECTOR, ".btnClose"),
        (By.CSS_SELECTOR, ".layerPop .close"),
    ]

    for by, value in candidates:
        try:
            elems = driver.find_elements(by, value)
            for elem in elems:
                if elem.is_displayed():
                    try:
                        elem.click()
                    except ElementClickInterceptedException:
                        driver.execute_script("arguments[0].click();", elem)
                    sleep_polite(0.1, 0.3)
        except Exception:
            pass


def manual_login(driver: Chrome) -> List[Dict]:
    login_url = "https://www.jobkorea.co.kr/login"
    print(f"[INFO] 로그인 페이지로 이동합니다: {login_url}")
    driver.get(login_url)
    
    print("\n" + "="*60)
    print("★ [수동 로그인 대기중] ★")
    print("열린 크롬 브라우저 창에서 잡코리아 로그인을 진행해 주세요.")
    print("로그인이 완료(로그아웃 버튼 활성화)되면 자동으로 세션을 복사해 병렬 크롤러를 구동합니다.")
    print("="*60 + "\n")
    
    max_wait = 300
    for i in range(max_wait):
        try:
            page_source = driver.page_source
            if "로그아웃" in page_source:
                print("[SUCCESS] 로그인이 감지되었습니다! 세션 정보를 복사하는 중...")
                sleep_polite(1.0, 1.5)
                cookies = driver.get_cookies()
                return cookies
        except Exception:
            pass
        time.sleep(1)
        if i % 10 == 0:
            print(f"[WAIT] 로그인 대기 중... ({i}/{max_wait}초)")
            
    print("[WARN] 로그인 대기 시간이 초과되었습니다. 일반 비로그인 세션으로 계속 진행합니다.")
    return []


def collect_links(
    driver: Chrome, start_page: int, end_page: int, output_path: Path
) -> List[str]:
    wait = WebDriverWait(driver, 15)
    links: List[str] = []
    seen = set()

    for page in range(start_page, end_page + 1):
        url = STARTER_URL.format(page=page)
        print(f"[LINK] page {page}: {url}")

        try:
            driver.get(url)
            close_possible_popups(driver)
            sleep_polite(0.5, 1.0)

            try:
                paper_list = wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH, "/html/body/div[4]/div[2]/div[2]/div[5]/ul")
                    )
                )
                a_tags = paper_list.find_elements(By.TAG_NAME, "a")
            except TimeoutException:
                a_tags = driver.find_elements(By.TAG_NAME, "a")

            for a in a_tags:
                href = a.get_attribute("href")
                if not href:
                    continue

                href_lower = href.lower()
                if "/starter/passassay/view" in href_lower or "passassay_view" in href_lower:
                    if "passassayindex" not in href_lower:
                        if href not in seen:
                            seen.add(href)
                            links.append(href)

        except WebDriverException as e:
            print(f"[WARN] page {page} failed: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(links), encoding="utf-8")
    print(f"[OK] collected links: {len(links)}")
    print(f"[OK] link file saved: {output_path}")
    return links


def expand_all_qna(driver: Chrome) -> None:
    try:
        # 모든 질문 펼치기 버튼을 JS로 즉시 일괄 강제 클릭하여 로딩 시간을 극적으로 줄입니다.
        driver.execute_script("document.querySelectorAll('.qnaLists dt button').forEach(btn => btn.click());")
        time.sleep(0.2)
    except Exception:
        pass


def crawl_detail(driver: Chrome, url: str) -> List[Dict[str, str]]:
    wait = WebDriverWait(driver, 10)
    rows: List[Dict[str, str]] = []

    clean_url = url.strip()
    if not clean_url:
        return rows

    try:
        driver.get(clean_url)
        close_possible_popups(driver)
        expand_all_qna(driver)

        user_info = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="container"]/div[2]/div[1]/div[1]/h2')
            )
        )

        company_el = find_one(user_info, By.TAG_NAME, "a")
        season_el = find_one(user_info, By.TAG_NAME, "em")

        company = text_or_empty(company_el)
        season = text_or_empty(season_el)

        spec_text = ""
        spec_el = find_one(driver, By.CLASS_NAME, "specLists")
        if spec_el:
            spec_items = [x.strip() for x in spec_el.text.split("\n") if x.strip()]
            spec_text = " | ".join(spec_items)

        paper = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "qnaLists")))

        question_elements = paper.find_elements(By.TAG_NAME, "dt")
        answer_elements = paper.find_elements(By.TAG_NAME, "dd")

        questions = []
        answers = []

        for q_el in question_elements:
            tx = find_one(q_el, By.CSS_SELECTOR, "span.tx")
            if tx is None:
                tx = find_one(q_el, By.CLASS_NAME, "tx")
            questions.append(text_or_empty(tx))

        for a_el in answer_elements:
            tx = find_one(a_el, By.CSS_SELECTOR, "div.tx")
            if tx is None:
                tx = find_one(a_el, By.CLASS_NAME, "tx")
            if tx is not None:
                try:
                    raw = driver.execute_script(
                        "return arguments[0].innerText;", tx
                    ) or ""
                    lines = [
                        line for line in raw.splitlines()
                        if not ("글자수" in line or "Byte" in line)
                    ]
                    answers.append("\n".join(lines).strip())
                except Exception:
                    answers.append(text_or_empty(tx))
            else:
                answers.append("")

        max_len = max(len(questions), len(answers))

        for i in range(max_len):
            q = questions[i] if i < len(questions) else ""
            a = answers[i] if i < len(answers) else ""

            if not q and not a:
                continue

            rows.append(
                {
                    "url": clean_url,
                    "company": company,
                    "season": season,
                    "specs": spec_text,
                    "question_no": str(i + 1),
                    "question": q,
                    "answer": a,
                }
            )

    except Exception as e:
        # 오류 상세 출력 방지로 깔끔한 로그 유지
        pass

    return rows


def append_to_csv_threadsafe(row: Dict[str, str], csv_path: Path) -> None:
    fieldnames = [
        "url",
        "company",
        "season",
        "specs",
        "question_no",
        "question",
        "answer",
    ]
    file_exists = csv_path.exists()

    with file_lock:
        try:
            with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists or csv_path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(row)
        except PermissionError:
            pass
        except Exception:
            pass


def append_to_jsonl_threadsafe(row: Dict[str, str], jsonl_path: Path) -> None:
    with file_lock:
        try:
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass


def chunk_list(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i::n] for i in range(n)]


def thread_crawler(
    thread_links: List[str],
    cookies: List[Dict],
    csv_path: Path,
    jsonl_path: Path,
    thread_idx: int,
    total_links_to_crawl: int,
    progress_counter: List[int],
) -> None:
    if not thread_links:
        print(f"[Thread {thread_idx}] 수집할 대상 링크가 없습니다.")
        return

    print(f"[Thread {thread_idx}] 크롤러 가동 시작 (수집 대상: {len(thread_links)}개)")
    driver = make_driver(headless=True)
    
    try:
        # 쿠키 주입을 위해 메인 도메인 우선 접속
        driver.get("https://www.jobkorea.co.kr/")
        time.sleep(1.0)
        
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception:
                pass
                
        # 세션 연동 대기
        driver.get("https://www.jobkorea.co.kr/")
        time.sleep(0.5)

        for idx, link in enumerate(thread_links, start=1):
            rows = crawl_detail(driver, link)
            
            if rows:
                for row in rows:
                    append_to_csv_threadsafe(row, csv_path)
                    append_to_jsonl_threadsafe(row, jsonl_path)
                
                # 메인 쓰레드 진행도 공유 및 카운트
                with file_lock:
                    progress_counter[0] += 1
                    current_cnt = progress_counter[0]
                
                print(f"[FAST CRAWL] [{current_cnt}/{total_links_to_crawl}] {row['company']} -> {len(rows)}개 문항 수집 완료!")
            
            sleep_polite(0.2, 0.4)

    except Exception as e:
        print(f"[Thread {thread_idx}] 에러 발생: {e}")
    finally:
        driver.quit()
        print(f"[Thread {thread_idx}] 크롤러 가동 종료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-login", action="store_true")
    parser.add_argument("--skip-link-crawl", action="store_true")
    parser.add_argument("--workers", type=int, default=4) # 기본 4개의 병렬 스레드 구동

    args = parser.parse_args()

    # 데이터 저장 경로 기본 설정
    data_dir = Path("./data")
    link_path = data_dir / "jobkorea_links.txt"
    csv_path = data_dir / "jobkorea_self_intro.csv"
    jsonl_path = data_dir / "jobkorea_self_intro.jsonl"

    print("="*60)
    print("★ 잡코리아 합격자소서 초고속 병렬 크롤러 ★")
    print(f" 병렬 가동 브라우저 수: {args.workers}대")
    print(f" 저장 파일: {csv_path.name}")
    print("="*60)

    # 1. 기존에 이미 수집 완료한 URL 리스트 파악하여 스킵(Skip) 준비
    crawled_urls = set()
    if csv_path.exists():
        try:
            print("[INFO] 기존에 이미 완료된 항목들을 스캔합니다...")
            with csv_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("url"):
                        crawled_urls.add(row["url"])
            print(f"[INFO] 이어받기 준비 완료: 이미 {len(crawled_urls)}개의 페이지가 수집되어 있습니다.")
        except Exception:
            pass

    # 2. 링크 수집 또는 파일 로드
    links = []
    if args.skip_link_crawl:
        links = []
        if link_path.exists():
            seen = set()
            for line in link_path.read_text(encoding="utf-8").splitlines():
                url = line.strip()
                if url and url not in seen:
                    seen.add(url)
                    links.append(url)
        print(f"[INFO] 파일에서 {len(links)}개의 전체 링크를 로드했습니다.")
    else:
        temp_driver = make_driver(headless=False)
        try:
            links = collect_links(
                driver=temp_driver,
                start_page=args.start_page,
                end_page=args.end_page,
                output_path=link_path,
            )
        finally:
            temp_driver.quit()

    if not links:
        print("[ERROR] 수집 대상 링크가 존재하지 않습니다.")
        return

    # 이미 수집된 URL 필터링 수행
    unvisited_links = [l for l in links if l not in crawled_urls]
    if args.limit is not None:
        unvisited_links = unvisited_links[:args.limit]

    print(f"[INFO] 최종 새로 수집해야 할 상세페이지 개수: {len(unvisited_links)}개")
    if not unvisited_links:
        print("[SUCCESS] 이미 모든 대상 상세페이지의 수집이 완료되어 있습니다!")
        return

    # 3. 로그인 정보(쿠키) 복사 단계
    cookies = []
    if not args.skip_login:
        login_driver = make_driver(headless=False)
        try:
            cookies = manual_login(login_driver)
        finally:
            login_driver.quit()
            print("[INFO] 수동 로그인 완료 및 세션 쿠키 추출이 완료되었습니다.")

    # 4. 링크들을 스레드 수만큼 등분
    chunks = chunk_list(unvisited_links, args.workers)
    progress_counter = [0]
    total_links_to_crawl = len(unvisited_links)

    print("\n" + "="*60)
    print(f"🚀 {args.workers}대의 headless 브라우저에 세션을 동기화하여 초고속 멀티 크롤링을 개시합니다!")
    print("="*60 + "\n")

    # 5. 병렬 스레드 가동
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(args.workers):
            f = executor.submit(
                thread_crawler,
                chunks[i],
                cookies,
                csv_path,
                jsonl_path,
                i + 1,
                total_links_to_crawl,
                progress_counter,
            )
            futures.append(f)
        
        # 모든 스레드가 완료될 때까지 대기
        for f in futures:
            f.result()

    print("\n" + "="*60)
    print("🎉 합격자소서 초고속 병렬 수집이 모두 종료되었습니다!")
    print(f" 최종 결과물: {csv_path.resolve()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
