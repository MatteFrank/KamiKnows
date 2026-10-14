"""Build MiniLM semantic chunks and PNG/PDF diagnostics; no retrieval or Qwen."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kamiknows.rag_ready.semantic_dataset import build_semantic_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--sentence-comparison-dir',type=Path)
    parser.add_argument('--target-words',type=int,default=400)
    parser.add_argument('--overlap-words',type=int,default=60)
    parser.add_argument('--min-words',type=int,default=250)
    parser.add_argument('--window-sentences',type=int,default=2)
    parser.add_argument('--threshold-quantile',type=float,default=0.25)
    parser.add_argument('--cut-policy',choices=['lowest_similarity','first_drop'],default='lowest_similarity',
                        help='First drop cuts at the earliest eligible significant decrease.')
    parser.add_argument('--min-score-drop',type=float,default=0.10,
                        help='Minimum cosine decrease for first_drop (default 0.10).')
    parser.add_argument('--exclude-backmatter',action='store_true',
                        help='Experimental: exclude acknowledgments and bibliography before chunking.')
    args=parser.parse_args()
    try:
        result=build_semantic_dataset(**vars(args))
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        print(f'Semantic build failed: {exc}',file=sys.stderr)
        return 1
    print(f"Papers: {result['papers_count']}; chunks: {result['chunks_v1_count']}")
    print(f"Lengths: {result['lengths']}; validation: {result['validation']}")
    print(f'Plots and diagnostics: {args.output_dir}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
