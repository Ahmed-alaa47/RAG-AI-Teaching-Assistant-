import re
import logging
import youtube_transcript_api
from typing import Optional, List

logger = logging.getLogger(__name__)

class YouTubeProcessor:
    """Handles extraction of transcripts from YouTube videos."""
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract the video ID from a YouTube URL."""
        # More targeted patterns to avoid matching protocol slashes
        patterns = [
            r'v=([0-9A-Za-z_-]{11})',           # ?v=ID or &v=ID
            r'youtu\.be\/([0-9A-Za-z_-]{11})',   # youtu.be/ID
            r'embed\/([0-9A-Za-z_-]{11})',      # youtube.com/embed/ID
            r'\/v\/([0-9A-Za-z_-]{11})'          # youtube.com/v/ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                logger.info(f"Regex found potential video ID: {video_id} using pattern: {pattern}")
                return video_id
        return None

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to H:M:S format."""
        from datetime import timedelta
        td = timedelta(seconds=int(seconds))
        # Ensure HH:MM:SS format
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_transcript(self, video_id: str) -> Optional[str]:
        """Fetch transcript for a given video ID."""
        try:
            # Instantiate the API client
            yt = youtube_transcript_api.YouTubeTranscriptApi()
            transcript_list = yt.list(video_id)
            
            # Try to find Arabic if possible, otherwise English
            try:
                transcript = transcript_list.find_transcript(['ar', 'en'])
            except:
                # Fallback to any available
                try:
                    transcript = transcript_list.find_generated_transcript(['ar', 'en'])
                except:
                    # Last resort: just get anything
                    transcript = next(iter(transcript_list))

            if transcript:
                data = transcript.fetch()
                # Format each segment with a timestamp [HH:MM:SS]
                formatted_segments = []
                for segment in data:
                    ts = self._format_timestamp(segment.start)
                    formatted_segments.append(f"[{ts}] {segment.text}")
                
                return " ".join(formatted_segments)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching YouTube transcript: {e}")
            return None

    def process_url(self, url: str) -> Optional[str]:
        """Complete pipeline: extract ID and get transcript."""
        video_id = self.extract_video_id(url)
        if not video_id:
            logger.warning(f"Could not extract video ID from URL: {url}")
            return None
        
        logger.info(f"Extracted YouTube ID: {video_id}. Fetching transcript...")
        return self.get_transcript(video_id)
