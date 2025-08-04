import requests
from bs4 import BeautifulSoup
import os
import shutil
from urllib.parse import urljoin, urlparse
import time
import re
from datetime import datetime, timedelta
from pathlib import Path

class PDFResearchCrawler:
    # 핵심 회사 6개로 정리 (실제 검색 가능한 회사명들)
    COMPANY_STOCK_MAP = {
        "삼성전자": "005930",
        "SK하이닉스": "000660", 
        "LG에너지솔루션": "373220",
        "현대차": "005380",
        "LG전자": "066570",
        "KIA": "000270",
    }
    
    def __init__(self, download_folder="pdf_downloads", max_downloads=3):
        self.download_folder = download_folder
        self.session = requests.Session()
        self.downloaded_count = 0
        self.max_downloads = max_downloads
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self._ensure_download_folder()
    
    def _ensure_download_folder(self):
        """다운로드 폴더 생성 및 정리"""
        if os.path.exists(self.download_folder):
            # 기존 폴더 내용 완전 삭제
            shutil.rmtree(self.download_folder)
        os.makedirs(self.download_folder)
        print(f"[폴더 정리] {self.download_folder} 폴더를 초기화했습니다.")
    
    def build_page_url(self, base_url, page_number):
        if '?' in base_url:
            if 'page=' in base_url:
                return re.sub(r'page=\d+', f'page={page_number}', base_url)
            else:
                return f"{base_url}&page={page_number}"
        else:
            return f"{base_url}?page={page_number}"
    
    def find_stock_items_by_title(self, soup, target_stock):
        try:
            stock_items = soup.find_all('a', class_='stock_item')
            print(f"페이지에서 찾은 stock_item 개수: {len(stock_items)}")
            
            # 디버깅: 실제 종목명들 확인
            found_stocks = []
            for item in stock_items[:5]:  # 처음 5개만 확인
                title = item.get('title', '')
                found_stocks.append(title)
            print(f"발견된 종목명들 (처음 5개): {found_stocks}")
            
            matched_items = []
            for item in stock_items:
                title = item.get('title', '')
                # 더 유연한 매칭 로직
                if (title.lower() == target_stock.lower() or 
                    target_stock.lower() in title.lower() or
                    title.lower() in target_stock.lower() or
                    # 특정 회사별 별칭 매칭
                    (target_stock == "카카오" and "카카오" in title) or
                    (target_stock == "SK하이닉스" and ("하이닉스" in title or "SK하이닉스" in title)) or
                    (target_stock == "삼성전자" and ("삼성전자" in title or "삼성" in title)) or
                    (target_stock == "현대차" and ("현대차" in title or "현대" in title))):
                    matched_items.append(item)
            
            print(f"'{target_stock}'와 일치하는 종목: {len(matched_items)}개")
            return matched_items
        except Exception as e:
            print(f"주식종목 검색 실패: {e}")
            return []
    
    def get_stock_filtered_pdf_links_from_page(self, url, target_stock):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            filtered_links = []
            matched_stock_items = self.find_stock_items_by_title(soup, target_stock)
            print(f"매칭된 종목 아이템 수: {len(matched_stock_items)}")
            
            for stock_item in matched_stock_items:
                row = stock_item.find_parent('tr')
                if row:
                    print(f"종목 '{target_stock}'의 행에서 PDF 링크 검색 중...")
                    pdf_links = self.find_pdf_links_in_row(row, target_stock)
                    print(f"찾은 PDF 링크 수: {len(pdf_links)}")
                    filtered_links.extend(pdf_links)
                else:
                    print(f"종목 '{target_stock}'의 부모 행을 찾을 수 없습니다.")
            
            print(f"총 필터링된 PDF 링크 수: {len(filtered_links)}")
            return filtered_links
        except requests.exceptions.RequestException as e:
            print(f"페이지 로딩 실패: {e}")
            return []
    
    def find_pdf_links_in_row(self, row, stock_name):
        pdf_links = []
        for link in row.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf') or 'pdf' in href.lower():
                full_url = urljoin("https://finance.naver.com", href)
                link_text = link.get_text(strip=True)
                filename = self.generate_filename(stock_name, link_text, full_url)
                pdf_links.append({
                    'url': full_url,
                    'filename': filename,
                    'text': link_text,
                    'stock': stock_name,
                    'row_content': row.get_text(strip=True)
                })
        return pdf_links
    
    def generate_filename(self, stock_name, link_text, url):
        url_filename = os.path.basename(urlparse(url).path)
        if not url_filename or not url_filename.endswith('.pdf'):
            url_filename = 'report.pdf'
        current_date = datetime.now().strftime("%Y%m%d")
        name_part, ext = os.path.splitext(url_filename)
        safe_stock_name = re.sub(r'[^\w\s-]', '', stock_name).strip()
        
        # link_text가 비어있으면 URL에서 파일명 추출
        if link_text and link_text.strip():
            report_name = re.sub(r'[^\w\s-]', '', link_text).strip()[:20]
        else:
            # URL에서 의미있는 부분 추출
            url_parts = url.split('/')
            for part in reversed(url_parts):
                if part and part != 'pdf' and not part.startswith('http'):
                    report_name = re.sub(r'[^\w\s-]', '', part).strip()[:15]
                    break
            else:
                report_name = "리포트"
        
        # 고유 ID 생성 (URL 해시 + 타임스탬프)
        url_hash = str(hash(url))[-6:]  # URL의 해시값 마지막 6자리
        timestamp = datetime.now().strftime("%H%M%S")
        unique_id = f"{url_hash}_{timestamp}"
        
        filename = f"{safe_stock_name}_{current_date}_{report_name}_{unique_id}{ext}"
        return filename
    
    def download_pdf(self, pdf_url, filename):
        try:
            print(f"다운로드 시작: {filename}")
            response = self.session.get(pdf_url, stream=True)
            response.raise_for_status()
            filepath = os.path.join(self.download_folder, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"다운로드 완료: {filepath}")
            self.downloaded_count += 1
            return True
        except requests.exceptions.RequestException as e:
            print(f"다운로드 실패 {filename}: {e}")
            return False
    
    def crawl_stock_reports(self, base_url, target_stock, max_pages=30):
        print(f"'{target_stock}' 종목의 리포트 크롤링 시작 (최대 {self.max_downloads}개)")
        all_pdf_links = []
        page = 1
        consecutive_empty_pages = 0  # 연속으로 빈 페이지가 나온 횟수
        
        # 1단계: 모든 페이지에서 PDF 링크 수집
        while page <= max_pages:
            page_url = self.build_page_url(base_url, page)
            print(f"페이지 {page} 크롤링: {page_url}")
            pdf_links = self.get_stock_filtered_pdf_links_from_page(page_url, target_stock)
            
            if pdf_links:
                print(f"페이지 {page}: {len(pdf_links)}개 PDF 파일 발견")
                all_pdf_links.extend(pdf_links)
                consecutive_empty_pages = 0  # 빈 페이지 카운터 리셋
            else:
                print(f"페이지 {page}: '{target_stock}' 종목의 리포트를 찾을 수 없습니다.")
                consecutive_empty_pages += 1
                
                # 연속 10페이지에서 아무것도 못 찾으면 중단 (더 많은 페이지를 검색하도록 개선)
                if consecutive_empty_pages >= 10:
                    print(f"연속 {consecutive_empty_pages}페이지에서 리포트를 찾지 못했습니다. 검색을 중단합니다.")
                    break
            
            page += 1
            time.sleep(1)
        # 2단계: 중복 제거 및 정렬
        unique_links = []
        seen_urls = set()
        for link in all_pdf_links:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])
        
        if not unique_links:
            print(f"'{target_stock}' 종목의 PDF 파일을 찾을 수 없습니다.")
            return 0
        
        print(f"'{target_stock}' 종목의 {len(unique_links)}개 PDF 파일을 찾았습니다.")
        for i, pdf in enumerate(unique_links, 1):
            print(f"  {i}. {pdf['text']}")
        
        # 3단계: 다운로드 실행
        print(f"\n다운로드 시작...")
        success_count = 0
        self.downloaded_count = 0  # 다운로드 카운터 초기화
        
        for i, pdf_info in enumerate(unique_links, 1):
            if self.downloaded_count >= self.max_downloads:
                print(f"최대 다운로드 개수({self.max_downloads})에 도달했습니다.")
                break
            print(f"\n[{i}/{len(unique_links)}]")
            if self.download_pdf(pdf_info['url'], pdf_info['filename']):
                success_count += 1
            time.sleep(1)
        
        print(f"\n크롤링 완료: {success_count}/{len(unique_links)} 파일 다운로드 성공")
        print(f"총 다운로드 개수: {self.downloaded_count}/{self.max_downloads}")
        return success_count
    
    def run_crawling(self, company_name: str) -> str:
        """메인 크롤링 실행 메서드"""
        try:
            stock_code = self.COMPANY_STOCK_MAP.get(company_name)
            if not stock_code:
                return f"[오류] '{company_name}'의 종목코드가 등록되어 있지 않습니다."
            
            base_url = "https://finance.naver.com/research/company_list.naver"
            success_count = self.crawl_stock_reports(base_url, company_name)
            
            if success_count > 0:
                return f"[PDF 크롤링 완료] {company_name} 리서치 리포트 {success_count}개 다운로드 완료"
            else:
                return f"[PDF 크롤링 실패] {company_name} 리서치 리포트를 찾을 수 없습니다."
        except Exception as e:
            return f"[PDF 크롤링 오류] {str(e)}"
    
    @classmethod
    def get_available_companies(cls) -> list:
        """사용 가능한 회사명 목록 반환"""
        return list(cls.COMPANY_STOCK_MAP.keys())
    
    @classmethod
    def validate_company(cls, company_name: str) -> bool:
        """회사명 유효성 검사"""
        return company_name in cls.COMPANY_STOCK_MAP
    
    @classmethod
    def get_stock_code(cls, company_name: str) -> str:
        """회사명으로 종목코드 반환"""
        return cls.COMPANY_STOCK_MAP.get(company_name, "")

if __name__ == "__main__":
    print("=== PDF 리서치 크롤러 테스트 ===")
    print("사용 가능한 회사명:")
    for company in PDFResearchCrawler.get_available_companies():
        print(f"  - {company}")
    print()
    
    # 테스트용 회사명 (SK하이닉스)
    test_company = "SK하이닉스"
    print(f"테스트 실행: {test_company}")
    
    # 클래스 인스턴스 생성 및 실행
    crawler = PDFResearchCrawler("pdf_downloads")
    result = crawler.run_crawling(test_company)
    print(result)
    
    # 사용자 입력 테스트 (선택사항)
    print("\n" + "="*50)
    user_input = input("다른 회사명으로 테스트하시겠습니까? (y/n): ")
    if user_input.lower() == 'y':
        company_name = input("크롤링할 회사명을 입력하세요: ")
        if crawler.validate_company(company_name):
            result = crawler.run_crawling(company_name)
            print(result)
        else:
            print(f"[오류] '{company_name}'는 지원되지 않는 회사명입니다.")
            print(f"지원되는 회사: {', '.join(crawler.get_available_companies())}") 
