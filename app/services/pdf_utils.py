"""
PDF Processing Service
Handles PDF text extraction, OCR, and document chunking
"""
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import io
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing including OCR and text extraction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            self.config = {}
        else:
            self.config = config
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.ocr_enabled = self.config.get('ocr_enabled', True)
        
        # Initialize OCR and PDF libraries
        self._initialize_libraries()
    
    def _initialize_libraries(self):
        """Initialize PDF and OCR libraries"""
        try:
            # Try to import PDF libraries
            global PyPDF2, fitz, pytesseract, pdf2image
            
            # For text-based PDFs
            try:
                import PyPDF2
                self.pypdf2_available = True
            except ImportError:
                self.pypdf2_available = False
                logger.warning("PyPDF2 not available")
            
            # For advanced PDF processing
            try:
                import fitz  # PyMuPDF
                self.pymupdf_available = True
            except ImportError:
                self.pymupdf_available = False
                logger.warning("PyMuPDF not available")
            
            # For OCR
            if self.ocr_enabled:
                try:
                    import pytesseract
                    from PIL import Image
                    import pdf2image
                    self.ocr_available = True
                except ImportError:
                    self.ocr_available = False
                    logger.warning("OCR libraries not available")
            
        except Exception as e:
            logger.error(f"Error initializing PDF libraries: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = None) -> Dict[str, Any]:
        """
        Extract text from PDF file
        Returns: Extracted text and metadata, including per-page offsets
        """
        print(f"PDFProcessor: Starting extraction for {pdf_path}")
        try:
            if not os.path.exists(pdf_path):
                print(f"PDFProcessor: File not found: {pdf_path}")
                return {"error": f"PDF file not found: {pdf_path}"}
            print(f"PDFProcessor: File exists, size: {os.path.getsize(pdf_path)} bytes")
            if use_ocr is None:
                use_ocr = self.ocr_enabled
            print(f"PDFProcessor: Available libraries - PyMuPDF: {self.pymupdf_available}, PyPDF2: {self.pypdf2_available}, OCR: {self.ocr_available}")
            # Try text-based extraction first
            print("PDFProcessor: Attempting text-based extraction...")
            text_result = self._extract_text_based(pdf_path)
            print(f"PDFProcessor: Text-based result: {text_result}")
            if text_result.get('success') and text_result.get('text', '').strip():
                # Add page_offsets: list of (start_char, end_char, page_number)
                page_offsets = []
                offset = 0
                for page in text_result.get('pages_info', []):
                    start = offset
                    end = offset + page['char_count']
                    page_offsets.append({'start': start, 'end': end, 'page_number': page['page_number']})
                    offset = end + 1  # +1 for the newline
                text_result['page_offsets'] = page_offsets
                return text_result
            # Fall back to OCR if text extraction failed or returned empty
            if use_ocr and self.ocr_available:
                print("PDFProcessor: Falling back to OCR extract")
                ocr_result = self._extract_text_ocr(pdf_path)
                if ocr_result.get('success'):
                    # Add page_offsets for OCR
                    page_offsets = []
                    offset = 0
                    for page in ocr_result.get('pages_info', []):
                        start = offset
                        end = offset + page['char_count']
                        page_offsets.append({'start': start, 'end': end, 'page_number': page['page_number']})
                        offset = end + 1
                    ocr_result['page_offsets'] = page_offsets
                return ocr_result
            print("PDFProcessor: No extraction method succeeded")
            return {"error": "Could not extract text from PDF"}
        except Exception as e:
            print(f"PDFProcessor: Exception in extract_text_from_pdf: {str(e)}")
            return {"error": f"PDF processing failed: {str(e)}"}
    
    def _extract_text_based(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from text-based PDF"""
        print("PDFProcessor: Starting text-based extraction")
        try:
            if self.pymupdf_available:
                print("PDFProcessor: Using PyMuPDF")
                return self._extract_with_pymupdf(pdf_path)
            elif self.pypdf2_available:
                print("PDFProcessor: Using PyPDF2")
                return self._extract_with_pypdf2(pdf_path)
            else:
                print("PDFProcessor: No PDF libraries available")
                return {"error": "No PDF text extraction library available"}
                
        except Exception as e:
            print(f"PDFProcessor: Text-based extraction exception: {str(e)}")
            return {"error": f"Text extraction failed: {str(e)}"}
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF"""
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            text = ""
            pages_info = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text + "\n"
                
                pages_info.append({
                    'page_number': page_num + 1,
                    'char_count': len(page_text),
                    'has_images': len(page.get_images()) > 0
                })
            
            doc.close()
            
            return {
                'success': True,
                'text': text,
                'pages': len(pages_info),
                'pages_info': pages_info,
                'method': 'pymupdf',
                'file_path': pdf_path
            }
            
        except Exception as e:
            return {"error": f"PyMuPDF extraction failed: {str(e)}"}
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyPDF2"""
        print("PDFProcessor: Starting PyPDF2 extraction")
        try:
            import PyPDF2
            print("PDFProcessor: PyPDF2 imported successfully")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"PDFProcessor: PDF has {len(pdf_reader.pages)} pages")
                text = ""
                pages_info = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    print(f"PDFProcessor: Page {page_num + 1} extracted {len(page_text)} characters")
                    
                    pages_info.append({
                        'page_number': page_num + 1,
                        'char_count': len(page_text)
                    })
                
                print(f"PDFProcessor: Total text extracted: {len(text)} characters")
                
                return {
                    'success': True,
                    'text': text,
                    'pages': len(pages_info),
                    'pages_info': pages_info,
                    'method': 'pypdf2',
                    'file_path': pdf_path
                }
                
        except Exception as e:
            print(f"PDFProcessor: PyPDF2 extraction failed: {str(e)}")
            return {"error": f"PyPDF2 extraction failed: {str(e)}"}
    
    def _extract_text_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using OCR"""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            text = ""
            pages_info = []
            
            for page_num, image in enumerate(images):
                # Extract text from image using OCR
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
                
                pages_info.append({
                    'page_number': page_num + 1,
                    'char_count': len(page_text),
                    'ocr_confidence': self._get_ocr_confidence(image)
                })
            
            return {
                'success': True,
                'text': text,
                'pages': len(pages_info),
                'pages_info': pages_info,
                'method': 'ocr',
                'file_path': pdf_path
            }
            
        except Exception as e:
            return {"error": f"OCR extraction failed: {str(e)}"}
    
    def _get_ocr_confidence(self, image) -> float:
        """Get OCR confidence score"""
        try:
            import pytesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            return sum(confidences) / len(confidences) if confidences else 0
        except:
            return 0.0
    
    def chunk_text(self, text: str, page_offsets: list = None, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks, assign page numbers if page_offsets provided
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        if not text.strip():
            return []
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_id = 0
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = current_chunk.strip()
                start_pos = len(''.join([c['text'] for c in chunks]))
                end_pos = start_pos + len(chunk_text)
                # Determine page(s) for this chunk
                pages = []
                if page_offsets:
                    for page in page_offsets:
                        if not (end_pos < page['start'] or start_pos > page['end']):
                            pages.append(page['page_number'])
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'size': current_size,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'hash': self._get_text_hash(chunk_text),
                    'page_numbers': pages if pages else None
                })
                if overlap > 0:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                    current_size = len(overlap_text) + sentence_size + 1
                else:
                    current_chunk = sentence
                    current_size = sentence_size
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size + (1 if current_chunk else 0)
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            start_pos = len(''.join([c['text'] for c in chunks]))
            end_pos = start_pos + len(chunk_text)
            pages = []
            if page_offsets:
                for page in page_offsets:
                    if not (end_pos < page['start'] or start_pos > page['end']):
                        pages.append(page['page_number'])
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'size': current_size,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'hash': self._get_text_hash(chunk_text),
                'page_numbers': pages if pages else None
            })
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - in production, use a proper NLP library
        import re
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text chunk"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def process_pdf_for_analysis(self, pdf_path: str, query: str = None) -> Dict[str, Any]:
        """
        Process PDF for analysis with chunking and metadata, including page numbers
        """
        logger.info(f"Processing PDF for analysis: {pdf_path}")
        try:
            extraction_result = self.extract_text_from_pdf(pdf_path)
            if not extraction_result.get('success'):
                return extraction_result
            text = extraction_result['text']
            page_offsets = extraction_result.get('page_offsets', None)
            chunks = self.chunk_text(text, page_offsets=page_offsets)
            relevant_chunks = []
            if query:
                relevant_chunks = self._find_relevant_chunks(chunks, query)
            return {
                'success': True,
                'file_path': pdf_path,
                'text': text,
                'text_length': len(text),
                'chunks': chunks,
                'chunk_count': len(chunks),
                'relevant_chunks': relevant_chunks,
                'pages': extraction_result.get('pages', 0),
                'extraction_method': extraction_result.get('method', 'unknown'),
                'metadata': {
                    'file_size': os.path.getsize(pdf_path),
                    'processed_at': 'timestamp_placeholder',
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }
            }
        except Exception as e:
            logger.error(f"Error processing PDF for analysis: {str(e)}")
            return {"error": f"PDF analysis processing failed: {str(e)}"}
    
    def _find_relevant_chunks(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Find chunks relevant to the query"""
        if not query:
            return []
        
        query_words = set(query.lower().split())
        relevant_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk['text'].lower()
            chunk_words = set(chunk_text.split())
            
            # Calculate relevance score
            overlap = len(query_words.intersection(chunk_words))
            relevance_score = overlap / len(query_words) if query_words else 0
            
            if relevance_score > 0.1:  # Threshold for relevance
                relevant_chunk = chunk.copy()
                relevant_chunk['relevance_score'] = relevance_score
                relevant_chunks.append(relevant_chunk)
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return relevant_chunks[:10]  # Return top 10 relevant chunks
    
    def extract_tables_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables from PDF"""
        logger.info(f"Extracting tables from PDF: {pdf_path}")
        
        try:
            # Placeholder for table extraction
            # In production, use libraries like tabula-py or camelot
            return {
                'success': True,
                'tables': [],
                'message': 'Table extraction not implemented yet'
            }
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return {"error": f"Table extraction failed: {str(e)}"}
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str = None) -> Dict[str, Any]:
        """Extract images from PDF"""
        logger.info(f"Extracting images from PDF: {pdf_path}")
        
        try:
            if not self.pymupdf_available:
                return {"error": "PyMuPDF required for image extraction"}
            
            import fitz
            
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        image_info = {
                            'page': page_num + 1,
                            'index': img_index,
                            'width': pix.width,
                            'height': pix.height,
                            'colorspace': pix.colorspace.name if pix.colorspace else 'unknown'
                        }
                        
                        if output_dir:
                            output_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index}.png")
                            pix.save(output_path)
                            image_info['saved_path'] = output_path
                        
                        images.append(image_info)
                    
                    pix = None
            
            doc.close()
            
            return {
                'success': True,
                'images': images,
                'image_count': len(images)
            }
            
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
            return {"error": f"Image extraction failed: {str(e)}"} 