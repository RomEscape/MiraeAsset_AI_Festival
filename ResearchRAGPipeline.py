import re
import fitz
import time
import uuid
import os
import http.client
import json
import chromadb
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel

class CLOVAStudioExecutor:
    def __init__(self, host, api_key):
        self._host = host
        self._api_key = api_key
        self.request_count = 0
        self.last_request_time = 0

    def _send_request(self, request, endpoint, max_retries=3):
        """API 호출 with 재시도 및 지연 시간 관리"""
        for attempt in range(max_retries):
            try:
                # Rate limiting: 요청 간 최소 3초 간격
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < 3.0:
                    sleep_time = 3.0 - time_since_last
                    time.sleep(sleep_time)
                
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self._api_key}',
                    'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4())
                }
                conn = http.client.HTTPSConnection(self._host)
                conn.request('POST', endpoint, json.dumps(request), headers)
                response = conn.getresponse()
                result = json.loads(response.read().decode('utf-8'))
                conn.close()
                
                self.request_count += 1
                self.last_request_time = time.time()
                
                # 성공 시 즉시 반환
                return result
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10 + random.uniform(1, 5)  # 지수 백오프 + 랜덤
                    print(f"429 에러 발생 → {wait_time:.1f}초 대기 후 재시도 (시도 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e
        
        raise Exception(f"최대 재시도 횟수({max_retries}) 초과")

class ResearchRAGPipeline:
    def __init__(self, db_path, collection_name):
        load_dotenv(override=True)
        self.embedding_model = ClovaXEmbeddings(model="bge-m3")
        self.client = chromadb.PersistentClient(path=db_path)
        # 기존 컬렉션 삭제 후 새로 생성
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # 컬렉션이 없으면 무시
        self.collection_name = collection_name
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )
        self.segmenter = CLOVAStudioExecutor(
            host="clovastudio.stream.ntruss.com",
            api_key=os.getenv("NCP_CLOVASTUDIO_API_KEY")
        )
        self.documents = []
        self.processed_files = set()  # 처리된 파일 추적

    def _extract_metadata_from_text(self, text: str):
        metadata = {
            "company": "",
            "date": "",
            "opinion": "",
            "analyst": "",
            "price_target": "",
            "importance_score": 0  # 중요도 점수 추가
        }
        
        # 회사명 추출
        company_match = re.search(r"([가-힣]+(?:전자|화학|바이오|에너지|솔루션|건설|증권|은행|제약))", text)
        if company_match:
            metadata["company"] = company_match.group(1)

        # 날짜 추출
        date_match = re.search(r"(\d{4}[.\-년 ]\d{1,2}[.\-월 ]\d{1,2})", text)
        if date_match:
            metadata["date"] = date_match.group(1).replace("년", "-").replace("월", "-").replace(" ", "-").replace(".", "-").strip("-")

        # 투자의견 추출
        opinion_match = re.search(r"(매수|중립|매도)", text)
        if opinion_match:
            metadata["opinion"] = opinion_match.group(1)

        # 애널리스트 추출
        analyst_match = re.search(r"(?:작성자|애널리스트)[\s:]*([가-힣]+)", text)
        if analyst_match:
            metadata["analyst"] = analyst_match.group(1)

        # 목표주가 추출
        price_match = re.search(r"목표주가[^\d]*(\d{2,3}(?:,\d{3})*(?:\.\d{1,2})?)", text)
        if price_match:
            metadata["price_target"] = price_match.group(1).replace(",", "")

        # 중요도 점수 계산
        importance_score = 0
        if metadata["opinion"]: importance_score += 10
        if metadata["price_target"]: importance_score += 15
        if metadata["analyst"]: importance_score += 5
        if len(text) > 1000: importance_score += 5  # 긴 문서는 더 중요
        if "삼성전자" in text: importance_score += 10  # 대상 종목 언급
        
        metadata["importance_score"] = importance_score
        return metadata

    def extract_from_pdf_folder(self, folder="./pdf_downloads"):
        path = Path(folder)
        if not path.exists() or not path.is_dir():
            raise ValueError("PDF 폴더 경로가 잘못되었습니다.")

        data_json = []
        merged_text = ""

        # PDF 파일들을 날짜 기준(최신순)으로 정렬
        pdf_files = []
        date_check_list = []  # 날짜 파싱 결과 확인용
        import re
        def extract_date_from_text(text):
            # 다양한 날짜 포맷 지원 (YYYYMMDD, YYYY-MM-DD, YYYY.MM.DD 등)
            patterns = [
                r'(20\\d{2})[.\\-년 ]?(\\d{1,2})[.\\-월 ]?(\\d{1,2})',  # 20250709, 2025-07-09, 2025.07.09, 2025년 7월 9일
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    y, mth, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
                    return f'{y}{mth}{d}'
            return None

        for file in path.glob("*.pdf"):
            if file.name in self.processed_files:
                continue  # 이미 처리된 파일 스킵
            # PDF 본문에서 날짜 추출
            import fitz
            text = ""
            doc = fitz.open(str(file))
            for page in doc:
                text += re.sub(r'\s+', ' ', page.get_text()).strip() + "\n"
            doc.close()
            date_str = extract_date_from_text(text)
            date_source = "본문"
            # 본문에서 추출 실패 시 파일명에서 추출
            if not date_str:
                m = re.search(r'_(\d{8})_', file.name)
                if m:
                    date_str = m.group(1)
                    date_source = "파일명"
                else:
                    date_str = "00000000"
                    date_source = "없음"
            pdf_files.append((file, date_str, text))
            date_check_list.append((file.name, date_str, date_source))
        # 날짜(YYYYMMDD) 기준 내림차순(최신순) 정렬
        pdf_files.sort(key=lambda x: x[1], reverse=True)
        # 날짜 파싱 결과 전체 출력
        print("[PDF 파일별 날짜 파싱 결과]")
        for fname, dstr, src in date_check_list:
            print(f"- {fname} → {dstr} (근거: {src})")

        for file, date_str, text in pdf_files:
            print(f"PDF 처리 중: {file.name} (날짜: {date_str})")
            metadata = self._extract_metadata_from_text(text)
            document = Document(
                page_content=text.strip(),
                metadata=metadata
            )
            self.documents.append(document)

            # JSON용 구조 추가
            data_json.append({
                "file_name": file.name,
                "content": text.strip(),
                "metadata": metadata
            })

            # 텍스트 파일 누적
            merged_text += f"\n\n[파일명: {file.name}]\n{text.strip()}"
            # 처리된 파일 기록
            self.processed_files.add(file.name)

        print(f"총 {len(self.documents)}개 PDF 문서가 로드되었습니다.")

        # 저장 경로 보장
        os.makedirs("./data", exist_ok=True)

        # JSON 저장
        with open("./data/research_reports.json", "w", encoding="utf-8") as f_json:
            json.dump(data_json, f_json, ensure_ascii=False, indent=2)
            print("research_reports.json 저장 완료")

        # TXT 저장
        with open("./data/merged_research_text.txt", "w", encoding="utf-8") as f_txt:
            f_txt.write(merged_text.strip())
            print("merged_research_text.txt 저장 완료")

    def segment_documents(self):
        if not self.documents:
            raise ValueError("PDF에서 문서를 먼저 추출해야 합니다. (self.documents가 비어 있음)")

        # 중요도 순으로 문서 정렬
        self.documents.sort(key=lambda x: x.metadata.get("importance_score", 0), reverse=True)
        
        segmented_docs = []
        batch_size = 2  # 배치 크기 줄임
        
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            print(f"세그멘테이션 배치 처리: {i+1}-{min(i+batch_size, len(self.documents))}/{len(self.documents)}")
            
            for doc in tqdm(batch, desc=f"배치 {i//batch_size + 1}"):
                try:
                    segments = self.segmenter._send_request(
                        request={
                            "postProcessMaxSize": 3000,  # 크기 줄임
                            "alpha": 0.0,
                            "segCnt": -1,
                            "postProcessMinSize": 1000,  # 최소 크기 줄임
                            "text": doc.page_content,
                            "postProcess": True
                        },
                        endpoint="/testapp/v1/api-tools/segmentation"
                    )["result"]["topicSeg"]

                    overlap = 1
                    for j in range(len(segments)):
                        seg_group = segments[j : j + 1 + overlap]
                        if not seg_group:
                            continue
                        combined = " ".join(" ".join(seg) if isinstance(seg, list) else seg for seg in seg_group)
                        segmented_docs.append(Document(
                            page_content=combined.strip(),
                            metadata=doc.metadata
                        ))
                        
                except Exception as e:
                    print(f"segmentation 실패 ({doc.metadata.get('file_name', 'unknown')}): {e}")
                    # 실패한 문서도 원본으로 추가
                    segmented_docs.append(doc)

            # 배치 간 지연
            if i + batch_size < len(self.documents):
                time.sleep(5)

        self.documents = segmented_docs
        print(f"총 {len(self.documents)}개의 segment 문서가 생성되었습니다.")

    def embed_and_store(self):
        if not self.documents:
            raise ValueError("segmentation 문서가 없습니다. 먼저 segment_documents()를 실행하세요.")

        texts, metadatas = [], []
        for i, doc in enumerate(self.documents):
            text = doc.page_content

            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)
            text = text.replace("\x00", "").strip()

            if not text:
                continue

            meta = doc.metadata
            flattened = {
                "id": str(uuid.uuid4()),
                "company": meta.get("company", ""),
                "opinion": meta.get("opinion", ""),
                "date": meta.get("date", ""),
                "analyst": meta.get("analyst", ""),
                "price_target": meta.get("price_target", ""),
                "importance_score": meta.get("importance_score", 0)
            }
            flattened = {k: v for k, v in flattened.items() if v}

            texts.append(text)
            metadatas.append(flattened)

        # 중요도 순으로 정렬
        sorted_data = sorted(zip(texts, metadatas), 
                           key=lambda x: x[1].get("importance_score", 0), reverse=True)
        texts, metadatas = zip(*sorted_data) if sorted_data else ([], [])

        # 배치 처리로 저장
        batch_size = 3  # 배치 크기 줄임
        success = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            print(f"임베딩 배치 처리: {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}")
            
            for j, (text, meta) in enumerate(tqdm(zip(batch_texts, batch_metadatas), 
                                                desc=f"배치 {i//batch_size + 1}")):
                try:
                    # 임베딩 생성
                    embedding = self.embedding_model.embed_documents([text])[0]
                    
                    # ChromaDB 저장
                    self.vectorstore.add_texts(
                        texts=[text],
                        metadatas=[meta],
                        ids=[f"doc_{i+j}_{meta.get('importance_score', 0)}"]
                    )
                    success += 1
                    
                except Exception as e:
                    if "429" in str(e):
                        print(f"429 에러 발생 → 30초 대기 후 재시도 (doc_{i+j})")
                        time.sleep(30.0)  # 30초 대기
                        try:
                            embedding = self.embedding_model.embed_documents([text])[0]
                            self.vectorstore.add_texts(
                                texts=[text],
                                metadatas=[meta],
                                ids=[f"doc_{i+j}_retry_{meta.get('importance_score', 0)}"]
                            )
                            success += 1
                        except Exception as e2:
                            print(f"재시도 실패 (doc_{i+j}): {e2}")
                            # 실패한 문서는 건너뛰고 계속 진행
                            continue
                    else:
                        print(f"저장 실패 (doc_{i+j}): {e}")
                        continue
                
                # 개별 문서 간 지연 (더 긴 대기 시간)
                time.sleep(3.0 + random.uniform(0, 2))
            
            # 배치 간 지연 (더 긴 대기 시간)
            if i + batch_size < len(texts):
                time.sleep(10.0 + random.uniform(0, 5))
            
            # 메모리 정리
            if i % 6 == 0:
                import gc
                gc.collect()

        print(f"\n총 {success}개 문서가 ChromaDB에 저장되었습니다.")

    def query(self, question: str) -> str:
        if self.vectorstore is None:
            raise ValueError("vector store가 초기화되지 않았습니다. 먼저 embed_and_store()를 실행하세요.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = PromptTemplate.from_template(
            '''당신은 금융 전문가이자 투자 심리 분석가입니다.
아래에 제공된 문맥(context)은 특정 종목에 대한 전문가 리서치 보고서입니다.
당신의 임무는 이 문맥을 기반으로 사용자의 질문에 대한 요약된 분석을 제공하는 것입니다.

답변 시 다음 지침을 따르세요:
- 정보의 출처가 명확하지 않으면 "정확한 정보가 확인되지 않습니다"라고 말하세요.
- 핵심 요점을 간결하게 전달하세요.
- 숫자나 회사명은 정확히 반복하고, 근거가 부족한 예측은 피하세요.
- 투자의견, 목표주가, 애널리스트 정보가 있으면 반드시 포함하세요.

⚠️ 반드시 실제 도구 실행 결과만 사용하세요. 예시를 복사하지 마세요.

# Question:
{question}

# Context:
{context}

# Answer:'''
        )

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content[:1500] for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | ChatClovaX(model="HCX-003", max_tokens=2048)
            | StrOutputParser()
        )

        rag_pipeline = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain)

        result = rag_pipeline.invoke(question)
        return result["answer"]


def main():
    rag = ResearchRAGPipeline(
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_research_docs"
    )
    rag.extract_from_pdf_folder("./pdf_downloads")
    rag.segment_documents()
    rag.embed_and_store()

    question = "삼성전자의 목표주가는 얼마인가요?"
    answer = rag.query(question)
    print(f"\n질문: {question}")
    print("응답:", answer)


if __name__ == "__main__":
    main()
