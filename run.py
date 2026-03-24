"""
Mavis launcher — run from repo root.
Usage:
    python run.py                          # pick interactively
    python run.py bicep                    # free mode
    python run.py shoulder                 # free mode
    python run.py bicep  --sets 3 --reps 10 --rest 60   # program mode
    python run.py shoulder --sets 4 --reps 8  --rest 90
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Mavis AI Fitness Coach")
    parser.add_argument("exercise", nargs="?", choices=["bicep", "shoulder"],
                        help="Exercise to run (bicep / shoulder)")
    parser.add_argument("--sets",  type=int, default=0,  help="Number of sets (0 = free mode)")
    parser.add_argument("--reps",  type=int, default=10, help="Reps per set")
    parser.add_argument("--rest",  type=int, default=60, help="Rest seconds between sets")
    args = parser.parse_args()

    exercise = args.exercise
    if not exercise:
        print("\nMAVIS — AI Fitness Coach")
        print("─" * 30)
        print("  1. Bicep Curl")
        print("  2. Shoulder Press")
        choice = input("\nSelect exercise (1/2): ").strip()
        exercise = "bicep" if choice == "1" else "shoulder"

    if exercise == "bicep":
        from biceps import BicepAnalyzer
        analyzer = BicepAnalyzer()
    else:
        from shoulders.shoulder_press import ShoulderAnalyzer
        analyzer = ShoulderAnalyzer()

    if args.sets > 0:
        analyzer.set_program(sets=args.sets, reps=args.reps, rest=args.rest)
        print(f"\n[Program] {args.sets} sets × {args.reps} reps | Rest: {args.rest}s")
    else:
        print("\n[Free mode] No set/rep target. Press Q to quit.")

    print("─" * 30)
    print("Starting in 1 second...\n")
    import time
    time.sleep(1)
    analyzer.run()


if __name__ == "__main__":
    main()
