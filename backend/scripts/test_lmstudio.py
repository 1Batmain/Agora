import os
from pipeline.claims.backend import resolve_backend
from pipeline.claims.ollama import OllamaStats

def test_lmstudio():
    print("Testing LM Studio backend...")
    # Surcharge optionnelle de l'URL si besoin, par defaut http://localhost:1234/v1
    backend = resolve_backend("lmstudio", model="mistral-7b-instruct-v0.3")
    
    print(f"Backend name: {backend.name}")
    print(f"Sovereign: {backend.sovereign}")
    
    messages = [
        {"role": "system", "content": "Tu es un assistant utile. Renvoie toujours une réponse au format JSON contenant une clé 'response'."},
        {"role": "user", "content": "Dis bonjour et donne un chiffre aléatoire."}
    ]
    
    stats = OllamaStats()
    print("Sending request...")
    result = backend.complete(messages, stats=stats)
    
    if result:
        print("\n=== Success ===")
        print(f"Result: {result}")
        print(f"Stats: calls={stats.calls}, errors={stats.errors}, cold_seconds={stats.cold_seconds:.2f}, eval_tokens={stats.eval_tokens}")
    else:
        print("\n=== Failed ===")
        print(f"Stats: errors={stats.errors}")

if __name__ == "__main__":
    test_lmstudio()
