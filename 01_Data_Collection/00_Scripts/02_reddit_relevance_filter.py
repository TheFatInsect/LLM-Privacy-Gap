"""
07_relevance_filter.py

Purpose: Filter Reddit posts to keep only those relevant to LLM privacy issues.
Strategy: Use LLM for zero-shot relevance classification with batching.

Pipeline:
1. Load raw crawled data (posts + comments)
2. Pre-filter: Remove posts with score <= 0
3. Batch posts and send to LLM for relevance evaluation
4. Keep relevant posts and their associated comments
5. Save cleaned dataset with statistics

Usage:
    python 07_relevance_filter.py [--input INPUT_CSV] [--batch-size N] [--test]
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

# Local utils (independent from 02_Qualitative_Analysis)
from utils.llm_client import LLMClient, LLMResponse

# =============================================================================
# Configuration
# =============================================================================

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_COLLECTION_DIR = BASE_DIR.parent
PROMPTS_DIR = DATA_COLLECTION_DIR / "00_Scripts" / "prompts"
OUTPUT_DIR = DATA_COLLECTION_DIR / "04_Data_Cleaning" / "outputs"
LOG_DIR = DATA_COLLECTION_DIR / "04_Data_Cleaning" / "logs"

# Default input file
DEFAULT_INPUT = DATA_COLLECTION_DIR / "02_Outputs" / "Crawled_Reddit" / "reddit_crawled.csv"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Session timestamp
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"relevance_filter_{SESSION_TS}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for the relevance filter."""
    min_score: int = 0                  # Minimum post score (filter out <= this)
    batch_size: int = 10                # Posts per LLM batch (larger = faster)
    max_workers: int = 70               # Parallel LLM requests (concurrent API calls)
    model: str = "openai/gpt-5"        # OrcaRouter model identifier
    temperature: float = 0.0            # Zero temp for deterministic output
    max_selftext_chars: int = 1000      # Truncate long posts (save tokens)
    confidence_threshold: float = 0.5   # Min confidence to trust
    test_mode: bool = False             # Process only first 50 posts
    include_examples: bool = True       # Include few-shot examples


# IGNORE_AUTHORS = {'AutoModerator', 'WithoutReason1729', '[deleted]', '[removed]', 'automoderator'}


@dataclass
class FilterStats:
    """Statistics from the filtering process."""
    total_posts_input: int = 0
    total_comments_input: int = 0
    # Pre-filtering (rule-based) - Posts
    posts_filtered_by_score: int = 0
    posts_filtered_no_human_comments: int = 0
    posts_filtered_empty_content: int = 0
    posts_filtered_deleted: int = 0
    posts_after_prefilter: int = 0
    # Pre-filtering (rule-based) - Comments (removed along with posts)
    comments_filtered_by_score: int = 0
    comments_filtered_no_human_comments: int = 0
    comments_filtered_empty_content: int = 0
    comments_filtered_deleted: int = 0
    comments_after_prefilter: int = 0
    # LLM filtering
    posts_sent_to_llm: int = 0
    posts_marked_relevant: int = 0
    posts_marked_irrelevant: int = 0
    posts_llm_failed: int = 0
    # Output
    comments_kept: int = 0
    comments_removed: int = 0
    total_llm_requests: int = 0
    total_tokens_used: int = 0
    elapsed_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "input": {
                "total_posts": self.total_posts_input,
                "total_comments": self.total_comments_input,
            },
            "prefilter_rule_based": {
                "posts_filtered_by_score": self.posts_filtered_by_score,
                "comments_filtered_by_score": self.comments_filtered_by_score,
                "posts_filtered_no_human_comments": self.posts_filtered_no_human_comments,
                "comments_filtered_no_human_comments": self.comments_filtered_no_human_comments,
                "posts_filtered_empty_content": self.posts_filtered_empty_content,
                "comments_filtered_empty_content": self.comments_filtered_empty_content,
                "posts_filtered_deleted": self.posts_filtered_deleted,
                "comments_filtered_deleted": self.comments_filtered_deleted,
                "posts_after_prefilter": self.posts_after_prefilter,
                "comments_after_prefilter": self.comments_after_prefilter,
            },
            "llm_filtering": {
                "posts_sent_to_llm": self.posts_sent_to_llm,
                "posts_marked_relevant": self.posts_marked_relevant,
                "posts_marked_irrelevant": self.posts_marked_irrelevant,
                "posts_llm_failed": self.posts_llm_failed,
            },
            "output": {
                "posts_kept": self.posts_marked_relevant,
                "comments_kept": self.comments_kept,
                "comments_removed": self.comments_removed,
            },
            "performance": {
                "total_llm_requests": self.total_llm_requests,
                "total_tokens_used": self.total_tokens_used,
                "elapsed_seconds": round(self.elapsed_seconds, 2),
            }
        }


# =============================================================================
# Prompt Loading
# =============================================================================

def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        logger.error(f"Prompt file not found: {filepath}")
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def build_system_prompt(include_examples: bool = True) -> str:
    """Build the complete system prompt with optional examples."""
    system = load_prompt("relevance_filter_system.txt")
    if include_examples:
        examples = load_prompt("relevance_filter_examples.txt")
        system = system + "\n\n" + examples
    return system


def build_user_prompt(posts: List[Dict], subreddit: str) -> str:
    """Build the user prompt for a batch of posts."""
    template = load_prompt("relevance_filter_user.txt")
    
    # Format posts content (compact format to save tokens)
    posts_content = []
    for i, post in enumerate(posts, 1):
        # Handle NaN/None selftext (pandas loads empty cells as float NaN)
        selftext = post.get('selftext', '')
        if pd.isna(selftext) or selftext is None:
            selftext_display = '[empty]'
        else:
            selftext_display = str(selftext)[:800] if selftext else '[empty]'  # Truncate to save tokens
        
        # Handle NaN title similarly
        title = post.get('title', '')
        if pd.isna(title) or title is None:
            title = '[no title]'
        
        # Compact format
        post_str = f"[{i}] id={post['id']} | title: {title} | selftext: {selftext_display}"
        posts_content.append(post_str)
    
    # Fill template
    prompt = template.replace("{batch_size}", str(len(posts)))
    prompt = prompt.replace("{subreddit}", subreddit)
    prompt = prompt.replace("{posts_content}", "\n".join(posts_content))
    
    return prompt


# =============================================================================
# LLM Relevance Evaluation
# =============================================================================

class RelevanceFilter:
    """LLM-based relevance filter for Reddit posts."""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.llm = LLMClient(
            model=config.model,
            temperature=config.temperature,
            max_tokens=2000,
            timeout=180.0,  # 3 minutes timeout for batch processing
        )
        self.system_prompt = build_system_prompt(config.include_examples)
        self.stats = FilterStats()
    
    def _get_posts_with_human_comments(self, posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> set:
        """
        Return set of post IDs that have at least one human comment.
        Human comment = comment not from AutoModerator, bots, or [deleted]/[removed]
        """
        # Get all unique post IDs
        all_post_ids = set(posts_df['id'].unique())
        
        # Filter comments to only human comments
        # Comments link to posts via 'link_id' field
        human_comments = comments_df[
            ~comments_df['body'].fillna('').str.lower().isin(['[deleted]', '[removed]', ''])
        ].copy()
        
        # Get post IDs that have at least one human comment
        posts_with_comments = set(human_comments['link_id'].unique())
        
        # Return intersection (posts that exist AND have human comments)
        return all_post_ids & posts_with_comments
        
    def evaluate_batch(self, posts: List[Dict], subreddit: str) -> Dict[str, Dict]:
        """
        Evaluate a batch of posts for relevance.
        
        Returns:
            Dict mapping post_id -> {"relevant": 0|1, "confidence": float, "reason": str}
        """
        user_prompt = build_user_prompt(posts, subreddit)
        
        results = {}
        
        try:
            response = self.llm.chat_completion(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                response_format="json"
            )
            
            self.stats.total_llm_requests += 1
            if response.usage:
                self.stats.total_tokens_used += response.usage.get("total_tokens", 0)
            
            if response.success and response.parsed_json:
                evaluations = response.parsed_json.get("evaluations", [])
                for eval_item in evaluations:
                    post_id = eval_item.get("post_id", "")
                    results[post_id] = {
                        "relevant": eval_item.get("relevant", 1),  # Default to relevant
                        "confidence": eval_item.get("confidence", 0.5),
                        "reason": eval_item.get("reason", "")
                    }
            else:
                # On failure, mark all as relevant (conservative)
                logger.warning(f"LLM batch failed: {response.error_message}")
                for post in posts:
                    results[post['id']] = {
                        "relevant": 1,
                        "confidence": 0.5,
                        "reason": "LLM evaluation failed - kept by default"
                    }
                    self.stats.posts_llm_failed += 1
                    
        except Exception as e:
            # Handle network/timeout errors gracefully
            logger.error(f"LLM request exception: {type(e).__name__}: {str(e)[:100]}")
            for post in posts:
                results[post['id']] = {
                    "relevant": 1,
                    "confidence": 0.5,
                    "reason": f"Exception: {type(e).__name__} - kept by default"
                }
                self.stats.posts_llm_failed += 1
        
        return results
    
    def process_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the full dataframe and return (cleaned_df, removed_df).
        
        Pre-filtering Pipeline (Rule-based):
        1. Filter posts with score <= 0
        2. Filter posts with no human comments (only bots/deleted)
        3. Filter posts with empty/deleted content
        4. Filter deleted/removed posts
        
        Then send remaining posts to LLM for relevance evaluation.
        """
        start_time = time.time()
        
        # Separate posts and comments
        posts_df = df[df['type'] == 'post'].copy()
        comments_df = df[df['type'] == 'comment'].copy()
        
        self.stats.total_posts_input = len(posts_df)
        self.stats.total_comments_input = len(comments_df)
        
        logger.info(f"Input: {self.stats.total_posts_input} posts, {self.stats.total_comments_input} comments")
        logger.info("="*50)
        logger.info("PHASE 1: Rule-based Pre-filtering")
        logger.info("="*50)
        
        # Track current comments (will be filtered along with posts)
        current_comments = comments_df.copy()
        
        # =====================================================================
        # STEP 1: Filter by score (score <= 0)
        # =====================================================================
        posts_step1 = posts_df[posts_df['score'] > self.config.min_score].copy()
        self.stats.posts_filtered_by_score = len(posts_df) - len(posts_step1)
        # Filter comments: keep only those linked to remaining posts
        comments_before = len(current_comments)
        current_comments = current_comments[current_comments['link_id'].isin(posts_step1['id'])]
        self.stats.comments_filtered_by_score = comments_before - len(current_comments)
        logger.info(f"[Step 1] Score > {self.config.min_score}: removed {self.stats.posts_filtered_by_score} posts + {self.stats.comments_filtered_by_score} comments, remaining {len(posts_step1)} posts")
        
        # =====================================================================
        # STEP 2: Filter posts with no human comments
        # A post needs at least 1 comment from a non-bot/non-deleted author
        # =====================================================================
        # Get all post IDs that have at least one human comment
        posts_with_human_comments = self._get_posts_with_human_comments(posts_step1, current_comments)
        posts_step2 = posts_step1[posts_step1['id'].isin(posts_with_human_comments)].copy()
        self.stats.posts_filtered_no_human_comments = len(posts_step1) - len(posts_step2)
        # Filter comments: keep only those linked to remaining posts
        comments_before = len(current_comments)
        current_comments = current_comments[current_comments['link_id'].isin(posts_step2['id'])]
        self.stats.comments_filtered_no_human_comments = comments_before - len(current_comments)
        logger.info(f"[Step 2] Has human comments: removed {self.stats.posts_filtered_no_human_comments} posts + {self.stats.comments_filtered_no_human_comments} comments, remaining {len(posts_step2)} posts")
        
        # =====================================================================
        # STEP 3: Filter posts with empty content (both title and selftext empty)
        # =====================================================================
        def has_content(row):
            title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
            selftext = str(row.get('selftext', '')) if pd.notna(row.get('selftext')) else ''
            # Remove common empty markers
            selftext = selftext.replace('[removed]', '').replace('[deleted]', '').strip()
            return len(title.strip()) > 0 or len(selftext) > 0
        
        posts_step3 = posts_step2[posts_step2.apply(has_content, axis=1)].copy()
        self.stats.posts_filtered_empty_content = len(posts_step2) - len(posts_step3)
        # Filter comments: keep only those linked to remaining posts
        comments_before = len(current_comments)
        current_comments = current_comments[current_comments['link_id'].isin(posts_step3['id'])]
        self.stats.comments_filtered_empty_content = comments_before - len(current_comments)
        logger.info(f"[Step 3] Has content: removed {self.stats.posts_filtered_empty_content} posts + {self.stats.comments_filtered_empty_content} comments, remaining {len(posts_step3)} posts")
        
        # =====================================================================
        # STEP 4: Filter deleted/removed posts (check title for [deleted]/[removed])
        # =====================================================================
        def is_not_deleted(row):
            title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
            return title.lower() not in ['[deleted]', '[removed]', 'deleted', 'removed']
        
        posts_step4 = posts_step3[posts_step3.apply(is_not_deleted, axis=1)].copy()
        self.stats.posts_filtered_deleted = len(posts_step3) - len(posts_step4)
        # Filter comments: keep only those linked to remaining posts
        comments_before = len(current_comments)
        current_comments = current_comments[current_comments['link_id'].isin(posts_step4['id'])]
        self.stats.comments_filtered_deleted = comments_before - len(current_comments)
        logger.info(f"[Step 4] Not deleted/removed: removed {self.stats.posts_filtered_deleted} posts + {self.stats.comments_filtered_deleted} comments, remaining {len(posts_step4)} posts")
        
        # =====================================================================
        # Pre-filter Summary
        # =====================================================================
        self.stats.posts_after_prefilter = len(posts_step4)
        self.stats.comments_after_prefilter = len(current_comments)
        total_posts_prefiltered = self.stats.total_posts_input - self.stats.posts_after_prefilter
        total_comments_prefiltered = self.stats.total_comments_input - self.stats.comments_after_prefilter
        logger.info(f"")
        logger.info(f"Pre-filter Summary:")
        logger.info(f"  Posts: {self.stats.total_posts_input} -> {self.stats.posts_after_prefilter} (removed {total_posts_prefiltered}, {100*total_posts_prefiltered/self.stats.total_posts_input:.1f}%)")
        logger.info(f"  Comments: {self.stats.total_comments_input} -> {self.stats.comments_after_prefilter} (removed {total_comments_prefiltered}, {100*total_comments_prefiltered/self.stats.total_comments_input:.1f}%)")
        
        # Use filtered posts and comments for LLM evaluation
        posts_for_llm = posts_step4.copy()
        comments_for_llm = current_comments.copy()
        
        # Test mode: limit posts
        if self.config.test_mode:
            posts_for_llm = posts_for_llm.head(50)
            logger.info(f"TEST MODE: Processing only {len(posts_for_llm)} posts")
        
        self.stats.posts_sent_to_llm = len(posts_for_llm)
        
        # =====================================================================
        # PHASE 2: LLM-based Relevance Filtering
        # =====================================================================
        logger.info("")
        logger.info("="*50)
        logger.info("PHASE 2: LLM-based Relevance Filtering")
        logger.info("="*50)
        logger.info("Press Ctrl+C to stop and save progress...")
        
        relevant_post_ids = set()
        removed_post_ids = set()
        evaluation_results = {}
        shutdown_triggered = False
        
        # Prepare all batches across all subreddits for parallel processing
        all_batches = []
        for subreddit, sub_posts in posts_for_llm.groupby('subreddit'):
            sub_posts_list = sub_posts.to_dict('records')
            for i in range(0, len(sub_posts_list), self.config.batch_size):
                batch = sub_posts_list[i:i + self.config.batch_size]
                all_batches.append((subreddit, batch))
        
        total_batches = len(all_batches)
        logger.info(f"Total batches to process: {total_batches} (using {self.config.max_workers} workers)")
        
        # Thread-safe containers
        import threading
        results_lock = threading.Lock()
        
        def process_batch(batch_info):
            """Process a single batch and return results."""
            subreddit, batch = batch_info
            return subreddit, batch, self.evaluate_batch(batch, subreddit)
        
        # Parallel processing with ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all batches
                futures = {executor.submit(process_batch, batch_info): batch_info 
                          for batch_info in all_batches}
                
                # Process results as they complete
                pbar = tqdm(as_completed(futures), total=total_batches, 
                           desc="Processing", unit="batch")
                for future in pbar:
                    if SHUTDOWN_REQUESTED:
                        logger.warning("Shutdown requested, cancelling remaining tasks...")
                        shutdown_triggered = True
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    try:
                        subreddit, batch, batch_results = future.result(timeout=300)
                        
                        with results_lock:
                            evaluation_results.update(batch_results)
                            
                            for post_id, result in batch_results.items():
                                if result['relevant'] == 1:
                                    relevant_post_ids.add(post_id)
                                    self.stats.posts_marked_relevant += 1
                                else:
                                    removed_post_ids.add(post_id)
                                    self.stats.posts_marked_irrelevant += 1
                        
                        # Update progress bar description
                        pbar.set_postfix(relevant=self.stats.posts_marked_relevant, 
                                        irrelevant=self.stats.posts_marked_irrelevant)
                                        
                    except Exception as e:
                        logger.error(f"Batch processing error: {type(e).__name__}: {str(e)[:100]}")
                        # Mark failed batch posts as relevant (conservative)
                        batch_info = futures[future]
                        _, batch = batch_info
                        with results_lock:
                            for post in batch:
                                evaluation_results[post['id']] = {
                                    "relevant": 1,
                                    "confidence": 0.5,
                                    "reason": f"Processing error - kept by default"
                                }
                                relevant_post_ids.add(post['id'])
                                self.stats.posts_marked_relevant += 1
                                self.stats.posts_llm_failed += 1
                    
        except KeyboardInterrupt:
            logger.warning("\n⚠️ KeyboardInterrupt received! Saving progress...")
            shutdown_triggered = True
        
        # =====================================================================
        # PHASE 3: Build Final Output
        # =====================================================================
        if shutdown_triggered:
            logger.info("")
            logger.info("="*50)
            logger.info("PARTIAL SAVE: Saving processed data so far...")
            logger.info("="*50)
        
        # Filter comments based on post relevance (from pre-filtered comments)
        # Comments belong to posts via 'link_id' field
        comments_kept = comments_for_llm[comments_for_llm['link_id'].isin(relevant_post_ids)]
        comments_removed_by_llm = comments_for_llm[~comments_for_llm['link_id'].isin(relevant_post_ids)]
        
        self.stats.comments_kept = len(comments_kept)
        self.stats.comments_removed = self.stats.total_comments_input - self.stats.comments_kept
        
        # Build final dataframes
        posts_kept = posts_for_llm[posts_for_llm['id'].isin(relevant_post_ids)]
        posts_removed = posts_df[~posts_df['id'].isin(relevant_post_ids)]
        
        cleaned_df = pd.concat([posts_kept, comments_kept], ignore_index=True)
        removed_df = pd.concat([posts_removed, comments_df[~comments_df['link_id'].isin(relevant_post_ids)]], ignore_index=True)
        
        self.stats.elapsed_seconds = time.time() - start_time
        
        status = "PARTIAL (interrupted)" if shutdown_triggered else "COMPLETE"
        logger.info(f"{status} in {self.stats.elapsed_seconds:.1f}s")
        logger.info(f"Kept: {len(posts_kept)} posts + {len(comments_kept)} comments = {len(cleaned_df)} total")
        logger.info(f"Removed: {len(posts_removed)} posts + {self.stats.comments_removed} comments = {len(removed_df)} total")
        
        return cleaned_df, removed_df, evaluation_results, shutdown_triggered


# =============================================================================
# Main Entry Point
# =============================================================================

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        logger.warning("Force quit requested. Exiting immediately...")
        sys.exit(1)
    logger.warning("\n⚠️ Ctrl+C detected! Finishing current batch and saving progress...")
    logger.warning("Press Ctrl+C again to force quit (data may be lost).")
    SHUTDOWN_REQUESTED = True


def main():
    global SHUTDOWN_REQUESTED
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Filter Reddit posts for LLM privacy relevance")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input CSV file")
    parser.add_argument("--batch-size", type=int, default=10, help="Posts per LLM batch (larger=faster)")
    parser.add_argument("--max-workers", type=int, default=70, help="Number of parallel LLM requests")
    parser.add_argument("--min-score", type=int, default=0, help="Minimum post score threshold")
    parser.add_argument("--model", type=str, default="gpt-5", help="LLM model to use")
    parser.add_argument("--test", action="store_true", help="Test mode (50 posts only)")
    parser.add_argument("--no-examples", action="store_true", help="Disable few-shot examples")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LLM Privacy Relevance Filter")
    logger.info("=" * 60)
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} total records")
    
    # Configure filter
    config = FilterConfig(
        min_score=args.min_score,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        model=args.model,
        test_mode=args.test,
        include_examples=not args.no_examples,
    )
    
    logger.info(f"Config: batch_size={config.batch_size}, max_workers={config.max_workers}, "
                f"min_score={config.min_score}, model={config.model}, test_mode={config.test_mode}")
    
    # Run filter
    filter_instance = RelevanceFilter(config)
    cleaned_df, removed_df, eval_results, was_interrupted = filter_instance.process_dataframe(df)
    
    # Save outputs
    output_suffix = "_test" if config.test_mode else ""
    if was_interrupted:
        output_suffix += "_partial"
    
    # 1. Cleaned data
    cleaned_path = OUTPUT_DIR / f"reddit_cleaned_{SESSION_TS}{output_suffix}.csv"
    cleaned_df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved cleaned data: {cleaned_path}")
    
    # 2. Removed data (for review)
    removed_path = OUTPUT_DIR / f"reddit_removed_{SESSION_TS}{output_suffix}.csv"
    removed_df.to_csv(removed_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved removed data: {removed_path}")
    
    # 3. Evaluation details
    eval_path = OUTPUT_DIR / f"evaluation_details_{SESSION_TS}{output_suffix}.json"
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved evaluation details: {eval_path}")
    
    # 4. Statistics
    stats_path = OUTPUT_DIR / f"filter_stats_{SESSION_TS}{output_suffix}.json"
    stats_dict = filter_instance.stats.to_dict()
    stats_dict["config"] = {
        "input_file": str(input_path),
        "batch_size": config.batch_size,
        "min_score": config.min_score,
        "model": config.model,
        "test_mode": config.test_mode,
        "was_interrupted": was_interrupted,
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved statistics: {stats_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("FILTER SUMMARY")
    print("=" * 70)
    print(f"Input Posts:                {filter_instance.stats.total_posts_input}")
    print(f"Input Comments:             {filter_instance.stats.total_comments_input}")
    print("-" * 70)
    print("Pre-filter (Rule-based):                    Posts    Comments")
    print(f"  - By Score (<=0):                         {filter_instance.stats.posts_filtered_by_score:>5}      {filter_instance.stats.comments_filtered_by_score:>6}")
    print(f"  - No Human Comments:                      {filter_instance.stats.posts_filtered_no_human_comments:>5}      {filter_instance.stats.comments_filtered_no_human_comments:>6}")
    print(f"  - Empty Content:                          {filter_instance.stats.posts_filtered_empty_content:>5}      {filter_instance.stats.comments_filtered_empty_content:>6}")
    print(f"  - Deleted/Removed:                        {filter_instance.stats.posts_filtered_deleted:>5}      {filter_instance.stats.comments_filtered_deleted:>6}")
    print(f"  After Pre-filter:                         {filter_instance.stats.posts_after_prefilter:>5}      {filter_instance.stats.comments_after_prefilter:>6}")
    print("-" * 70)
    print("LLM Filtering:")
    print(f"  Sent to LLM:              {filter_instance.stats.posts_sent_to_llm}")
    print(f"  Marked Relevant:          {filter_instance.stats.posts_marked_relevant}")
    print(f"  Marked Irrelevant:        {filter_instance.stats.posts_marked_irrelevant}")
    print(f"  LLM Failures:             {filter_instance.stats.posts_llm_failed}")
    print("-" * 70)
    print("Final Output:")
    print(f"  Posts Kept:               {filter_instance.stats.posts_marked_relevant}")
    print(f"  Comments Kept:            {filter_instance.stats.comments_kept}")
    print(f"  Total Tokens Used:        {filter_instance.stats.total_tokens_used}")
    print(f"  Elapsed Time:             {filter_instance.stats.elapsed_seconds:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
