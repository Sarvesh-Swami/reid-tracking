"""
List Available Gemini Models
=============================
Shows which Gemini models are available with your API key.

Usage:
    python list_gemini_models.py --api-key "your_key_here"
"""

import argparse

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai not installed")
    print("Install with: pip install google-generativeai")
    exit(1)


def list_models(api_key):
    """List all available Gemini models."""
    genai.configure(api_key=api_key)
    
    print("\n" + "="*60)
    print("AVAILABLE GEMINI MODELS")
    print("="*60)
    
    try:
        models = genai.list_models()
        
        print("\nModels that support generateContent:")
        print("-" * 60)
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"\n✓ {model.name}")
                print(f"  Display Name: {model.display_name}")
                print(f"  Description: {model.description[:100]}...")
                print(f"  Input Token Limit: {model.input_token_limit}")
                print(f"  Output Token Limit: {model.output_token_limit}")
        
        print("\n" + "="*60)
        print("\nRECOMMENDED MODELS FOR TRACKING:")
        print("-" * 60)
        print("1. gemini-1.5-flash-latest (Fast, good for video)")
        print("2. gemini-1.5-pro-latest (Accurate, slower)")
        print("3. gemini-pro (Stable, widely available)")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is correct")
        print("2. Verify API is enabled in Google Cloud Console")
        print("3. Check you haven't exceeded quota")


def main():
    parser = argparse.ArgumentParser(description='List Available Gemini Models')
    parser.add_argument('--api-key', type=str, required=True, help='Gemini API key')
    args = parser.parse_args()
    
    list_models(args.api_key)


if __name__ == '__main__':
    main()
