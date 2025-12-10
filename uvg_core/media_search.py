"""
UVG MAX Media Search Module

Multi-provider stock media search with:
- HEAD pre-filter (skip large files before download)
- ThreadPoolExecutor for parallel downloads
- Retry with exponential backoff
- Provider adapters for Pexels, Pixabay, Unsplash, Coverr, Archive.org
"""

import os
import logging
import hashlib
import time
import json
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class MediaCandidate:
    """A media candidate from search."""
    provider: str
    id: str
    url: str
    download_url: str
    content_length: int = 0  # bytes
    duration: float = 0.0  # seconds
    width: int = 0
    height: int = 0
    thumbnail_url: str = ""
    title: str = ""
    downloaded_path: str = ""
    sha256: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "id": self.id,
            "url": self.url,
            "download_url": self.download_url,
            "content_length": self.content_length,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "thumbnail_url": self.thumbnail_url,
            "title": self.title,
            "downloaded_path": self.downloaded_path,
            "sha256": self.sha256,
        }


@dataclass
class SearchResult:
    """Result of a media search."""
    query: str
    candidates: List[MediaCandidate] = field(default_factory=list)
    total_found: int = 0
    providers_queried: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# PROVIDER ADAPTERS
# =============================================================================

class ProviderAdapter:
    """Base class for provider adapters."""
    
    name: str = "base"
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
    
    def search(self, query: str, max_results: int = 10) -> List[MediaCandidate]:
        raise NotImplementedError
    
    def is_available(self) -> bool:
        return bool(self.api_key)


class PexelsAdapter(ProviderAdapter):
    """Pexels video search adapter."""
    
    name = "pexels"
    BASE_URL = "https://api.pexels.com/videos/search"
    
    def search(self, query: str, max_results: int = 10) -> List[MediaCandidate]:
        if not self.api_key:
            return []
        
        try:
            headers = {"Authorization": self.api_key}
            params = {"query": query, "per_page": min(max_results, 80)}
            
            response = requests.get(
                self.BASE_URL, 
                headers=headers, 
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            candidates = []
            for video in data.get("videos", []):
                # Get best quality video file
                video_files = video.get("video_files", [])
                if not video_files:
                    continue
                
                # Sort by quality (width)
                video_files.sort(key=lambda x: x.get("width", 0), reverse=True)
                best = video_files[0]
                
                candidates.append(MediaCandidate(
                    provider=self.name,
                    id=str(video.get("id")),
                    url=video.get("url", ""),
                    download_url=best.get("link", ""),
                    content_length=best.get("file_size", 0) or 0,
                    duration=video.get("duration", 0),
                    width=best.get("width", 0),
                    height=best.get("height", 0),
                    thumbnail_url=video.get("image", ""),
                    title=video.get("url", "").split("/")[-1],
                ))
            
            return candidates
            
        except Exception as e:
            logger.warning(f"Pexels search failed: {e}")
            return []


class PixabayAdapter(ProviderAdapter):
    """Pixabay video search adapter."""
    
    name = "pixabay"
    BASE_URL = "https://pixabay.com/api/videos/"
    
    def search(self, query: str, max_results: int = 10) -> List[MediaCandidate]:
        if not self.api_key:
            return []
        
        try:
            params = {
                "key": self.api_key,
                "q": query,
                "per_page": min(max_results, 200),
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            candidates = []
            for video in data.get("hits", []):
                videos = video.get("videos", {})
                # Prefer large, then medium
                best = videos.get("large", videos.get("medium", {}))
                
                if not best:
                    continue
                
                candidates.append(MediaCandidate(
                    provider=self.name,
                    id=str(video.get("id")),
                    url=video.get("pageURL", ""),
                    download_url=best.get("url", ""),
                    content_length=best.get("size", 0),
                    duration=video.get("duration", 0),
                    width=best.get("width", 0),
                    height=best.get("height", 0),
                    thumbnail_url=video.get("userImageURL", ""),
                    title=video.get("tags", "")[:50],
                ))
            
            return candidates
            
        except Exception as e:
            logger.warning(f"Pixabay search failed: {e}")
            return []


class UnsplashAdapter(ProviderAdapter):
    """Unsplash image search (images→video via motion)."""
    
    name = "unsplash"
    BASE_URL = "https://api.unsplash.com/search/photos"
    
    def search(self, query: str, max_results: int = 10) -> List[MediaCandidate]:
        if not self.api_key:
            return []
        
        try:
            headers = {"Authorization": f"Client-ID {self.api_key}"}
            params = {"query": query, "per_page": min(max_results, 30)}
            
            response = requests.get(
                self.BASE_URL,
                headers=headers,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            candidates = []
            for photo in data.get("results", []):
                urls = photo.get("urls", {})
                
                candidates.append(MediaCandidate(
                    provider=self.name,
                    id=photo.get("id", ""),
                    url=photo.get("links", {}).get("html", ""),
                    download_url=urls.get("full", urls.get("regular", "")),
                    width=photo.get("width", 0),
                    height=photo.get("height", 0),
                    thumbnail_url=urls.get("thumb", ""),
                    title=photo.get("description", "")[:50] if photo.get("description") else "",
                    duration=0,  # Images have no duration
                ))
            
            return candidates
            
        except Exception as e:
            logger.warning(f"Unsplash search failed: {e}")
            return []


class LocalAdapter(ProviderAdapter):
    """Local file system adapter for test mode."""
    
    name = "local"
    
    def __init__(self, assets_dir: str = "./assets"):
        super().__init__("")
        self.assets_dir = Path(assets_dir)
    
    def is_available(self) -> bool:
        return self.assets_dir.exists()
    
    def search(self, query: str, max_results: int = 10) -> List[MediaCandidate]:
        if not self.assets_dir.exists():
            return []
        
        candidates = []
        extensions = [".mp4", ".mov", ".webm", ".avi", ".jpg", ".png"]
        
        for ext in extensions:
            for file_path in self.assets_dir.rglob(f"*{ext}"):
                if len(candidates) >= max_results:
                    break
                
                try:
                    stat = file_path.stat()
                    candidates.append(MediaCandidate(
                        provider=self.name,
                        id=file_path.stem,
                        url=str(file_path),
                        download_url=str(file_path),
                        content_length=stat.st_size,
                        title=file_path.stem,
                        downloaded_path=str(file_path),
                    ))
                except Exception:
                    pass
        
        return candidates


# =============================================================================
# MEDIA SEARCH ENGINE
# =============================================================================

class MediaSearchEngine:
    """
    Multi-provider media search with intelligent filtering.
    
    Features:
    - HEAD request pre-filter (skip before download)
    - Parallel downloads with ThreadPoolExecutor
    - Retry with exponential backoff
    - Size and duration filtering
    """
    
    MAX_WORKERS = 6
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 clips_dir: Optional[Path] = None,
                 max_clip_size_mb: int = 120,
                 min_duration: float = 1.0,
                 max_duration: float = 120.0,
                 local_test_mode: bool = False):
        """
        Initialize media search engine.
        
        Args:
            config: Configuration dict with API keys
            clips_dir: Directory to save downloaded clips
            max_clip_size_mb: Maximum clip size in MB
            min_duration: Minimum clip duration
            max_duration: Maximum clip duration
            local_test_mode: Use local assets only
        """
        self.config = config or {}
        self.clips_dir = Path(clips_dir) if clips_dir else Path("./uvg_output/clips")
        self.max_clip_size_bytes = max_clip_size_mb * 1024 * 1024
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.local_test_mode = local_test_mode
        
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize adapters
        self.adapters: List[ProviderAdapter] = []
        self._init_adapters()
        
        # Download log
        self.download_log: List[Dict] = []
    
    def _init_adapters(self) -> None:
        """Initialize provider adapters from config or environment."""
        if self.local_test_mode:
            self.adapters.append(LocalAdapter(
                self.config.get("assets_dir", "./assets")
            ))
            return
        
        # Load from dotenv if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Get keys from config or environment
        pexels_key = self.config.get("PEXELS_KEY") or os.getenv("PEXELS_KEY", "")
        pixabay_key = self.config.get("PIXABAY_KEY") or os.getenv("PIXABAY_KEY", "")
        unsplash_key = self.config.get("UNSPLASH_KEY") or os.getenv("UNSPLASH_ACCESS_KEY", "")
        coverr_key = self.config.get("COVERR_KEY") or os.getenv("COVERR_KEY", "")
        
        # Add providers in priority order
        if pexels_key:
            self.adapters.append(PexelsAdapter(pexels_key))
            logger.info("Pexels adapter initialized")
        
        if pixabay_key:
            self.adapters.append(PixabayAdapter(pixabay_key))
            logger.info("Pixabay adapter initialized")
        
        if unsplash_key:
            self.adapters.append(UnsplashAdapter(unsplash_key))
            logger.info("Unsplash adapter initialized")
        
        # Always add local as fallback
        self.adapters.append(LocalAdapter(
            self.config.get("assets_dir", "./assets")
        ))
        
        logger.info(f"Initialized {len(self.adapters)} media providers")
    
    def _head_check(self, url: str) -> Tuple[bool, int]:
        """
        Check file size via HEAD request before downloading.
        
        Args:
            url: URL to check
            
        Returns:
            (should_download, content_length)
        """
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            
            content_length = int(response.headers.get("Content-Length", 0))
            
            # Skip if too large
            if content_length > self.max_clip_size_bytes:
                logger.debug(f"Skipping {url}: size {content_length} > max {self.max_clip_size_bytes}")
                return False, content_length
            
            return True, content_length
            
        except Exception as e:
            logger.debug(f"HEAD check failed for {url}: {e}")
            # If HEAD fails, try downloading anyway
            return True, 0
    
    def _download_with_retry(self, 
                              candidate: MediaCandidate,
                              retries: int = 3) -> Optional[str]:
        """
        Download a file with retry logic.
        
        Args:
            candidate: Media candidate to download
            retries: Number of retries
            
        Returns:
            Downloaded file path or None
        """
        url = candidate.download_url
        
        for attempt in range(retries):
            try:
                # HEAD check first
                should_download, size = self._head_check(url)
                if not should_download:
                    self._log_download(candidate, False, "size_exceeded")
                    return None
                
                # Download
                response = requests.get(url, timeout=self.DEFAULT_TIMEOUT, stream=True)
                response.raise_for_status()
                
                # Generate filename
                ext = self._get_extension(url, response)
                filename = f"{candidate.provider}_{candidate.id}{ext}"
                filepath = self.clips_dir / filename
                
                # Stream to file with size limit
                downloaded = 0
                sha256 = hashlib.sha256()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        
                        # Abort if exceeding max size
                        if downloaded > self.max_clip_size_bytes:
                            f.close()
                            filepath.unlink(missing_ok=True)
                            self._log_download(candidate, False, "size_exceeded_during_download")
                            return None
                        
                        f.write(chunk)
                        sha256.update(chunk)
                
                candidate.downloaded_path = str(filepath)
                candidate.sha256 = sha256.hexdigest()
                candidate.content_length = downloaded
                
                self._log_download(candidate, True, "success")
                logger.debug(f"Downloaded {filepath} ({downloaded} bytes)")
                
                return str(filepath)
                
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt  # Exponential backoff
                logger.warning(f"Download attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        
        self._log_download(candidate, False, "max_retries_exceeded")
        return None
    
    def _get_extension(self, url: str, response: requests.Response) -> str:
        """Get file extension from URL or content type."""
        # Try URL first
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        for ext in ['.mp4', '.mov', '.webm', '.avi', '.jpg', '.png', '.jpeg']:
            if path.endswith(ext):
                return ext
        
        # Try content type
        content_type = response.headers.get("Content-Type", "")
        type_map = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/webm": ".webm",
            "image/jpeg": ".jpg",
            "image/png": ".png",
        }
        
        for ct, ext in type_map.items():
            if ct in content_type:
                return ext
        
        return ".mp4"  # Default
    
    def _log_download(self, candidate: MediaCandidate, success: bool, reason: str) -> None:
        """Log download attempt."""
        self.download_log.append({
            "provider": candidate.provider,
            "id": candidate.id,
            "url": candidate.download_url,
            "success": success,
            "reason": reason,
            "timestamp": time.time(),
        })
    
    def search(self, query: str, max_results: int = 20) -> SearchResult:
        """
        Search all providers in parallel.
        
        Args:
            query: Search query
            max_results: Maximum results per provider
            
        Returns:
            SearchResult with candidates
        """
        result = SearchResult(query=query)
        
        def search_provider(adapter: ProviderAdapter) -> Tuple[str, List[MediaCandidate]]:
            try:
                candidates = adapter.search(query, max_results)
                return adapter.name, candidates
            except Exception as e:
                logger.warning(f"Provider {adapter.name} failed: {e}")
                return adapter.name, []
        
        # Search in parallel
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {
                executor.submit(search_provider, adapter): adapter 
                for adapter in self.adapters if adapter.is_available()
            }
            
            for future in as_completed(futures):
                provider_name, candidates = future.result()
                result.providers_queried.append(provider_name)
                result.candidates.extend(candidates)
        
        # Filter by duration
        result.candidates = [
            c for c in result.candidates
            if c.duration == 0  # Images have no duration
            or self.min_duration <= c.duration <= self.max_duration
        ]
        
        result.total_found = len(result.candidates)
        logger.info(f"Found {result.total_found} candidates for '{query}'")
        
        return result
    
    def download_candidates(self, 
                             candidates: List[MediaCandidate],
                             max_downloads: int = 10) -> List[MediaCandidate]:
        """
        Download candidates in parallel.
        
        Args:
            candidates: Candidates to download
            max_downloads: Maximum to download
            
        Returns:
            Successfully downloaded candidates
        """
        to_download = candidates[:max_downloads]
        downloaded = []
        
        def download_one(candidate: MediaCandidate) -> Optional[MediaCandidate]:
            path = self._download_with_retry(candidate)
            if path:
                return candidate
            return None
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_one, c): c 
                for c in to_download
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    downloaded.append(result)
        
        logger.info(f"Downloaded {len(downloaded)}/{len(to_download)} candidates")
        
        return downloaded
    
    def search_and_download(self, 
                             query: str,
                             max_candidates: int = 20,
                             max_downloads: int = 10) -> List[MediaCandidate]:
        """
        Search and download in one step.
        
        Args:
            query: Search query
            max_candidates: Max search results
            max_downloads: Max downloads
            
        Returns:
            Downloaded candidates
        """
        result = self.search(query, max_candidates)
        return self.download_candidates(result.candidates, max_downloads)
    
    def get_download_log(self) -> List[Dict]:
        """Get download audit log."""
        return self.download_log
    
    def save_log(self, path: Path) -> None:
        """Save download log to file."""
        with open(path, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def rank_with_selector(self,
                           candidates: List[MediaCandidate],
                           prompt: str,
                           top_k: int = 5) -> List[MediaCandidate]:
        """
        Rank downloaded candidates using uvg_selector.
        
        Args:
            candidates: Downloaded candidates with downloaded_path set
            prompt: Text prompt for semantic matching
            top_k: Number of top candidates to return
            
        Returns:
            Reordered list of candidates by relevance
        """
        # Filter to only downloaded candidates
        downloaded = [c for c in candidates if c.downloaded_path]
        
        if not downloaded:
            return candidates
        
        try:
            from uvg_selector.clip_selector import rank_clips
            
            # Get paths for ranking
            clip_paths = [c.downloaded_path for c in downloaded]
            
            # Rank with selector
            ranked = rank_clips(prompt, clip_paths, top_k=top_k)
            
            # Reorder candidates based on ranking
            path_to_candidate = {c.downloaded_path: c for c in downloaded}
            reordered = []
            
            for r in ranked:
                path = r["path"]
                if path in path_to_candidate:
                    candidate = path_to_candidate[path]
                    # Store selector score in candidate
                    candidate.title = f"{candidate.title} [score: {r.get('final_score', 0):.2f}]"
                    reordered.append(candidate)
            
            # Add any remaining candidates not in ranking
            ranked_paths = {r["path"] for r in ranked}
            for c in downloaded:
                if c.downloaded_path not in ranked_paths:
                    reordered.append(c)
            
            logger.info(f"Ranked {len(reordered)} candidates with uvg_selector")
            return reordered
            
        except ImportError:
            logger.warning("uvg_selector not available, returning original order")
            return candidates
        except Exception as e:
            logger.warning(f"Selector ranking failed: {e}")
            return candidates
    
    def search_download_and_rank(self,
                                  query: str,
                                  max_candidates: int = 20,
                                  max_downloads: int = 10,
                                  top_k: int = 5) -> List[MediaCandidate]:
        """
        Search, download, and rank with uvg_selector.
        
        Args:
            query: Search query (also used as prompt for ranking)
            max_candidates: Max search results
            max_downloads: Max downloads
            top_k: Top K to return after ranking
            
        Returns:
            Top-ranked downloaded candidates
        """
        # Search and download
        downloaded = self.search_and_download(query, max_candidates, max_downloads)
        
        # Rank with selector
        ranked = self.rank_with_selector(downloaded, query, top_k)
        
        return ranked


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def search_media(query: str, 
                 max_results: int = 20,
                 config: Optional[Dict] = None) -> SearchResult:
    """Search for media across providers."""
    engine = MediaSearchEngine(config=config)
    return engine.search(query, max_results)


def download_media(candidates: List[MediaCandidate],
                   clips_dir: Optional[Path] = None) -> List[MediaCandidate]:
    """Download media candidates."""
    engine = MediaSearchEngine(clips_dir=clips_dir)
    return engine.download_candidates(candidates)
