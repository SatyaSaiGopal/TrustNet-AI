/**
 * TrustNet AI — API Client Module
 * Handles all network communication with the FastAPI backend.
 */

const API_BASE = "";

class TrustNetAPI {
  /**
   * Check connection status to the backend.
   */
  async getStatus() {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) {
      throw new Error(`Backend unavailable (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Get installed Ollama models and active selection.
   */
  async getModels() {
    const res = await fetch(`${API_BASE}/api/models`);
    if (!res.ok) {
      throw new Error(`Failed to load models (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Switch active LLM model.
   */
  async selectModel(model) {
    const res = await fetch(`${API_BASE}/api/models/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model })
    });
    if (!res.ok) {
      throw new Error(`Failed to set model (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Fetch synthetic demonstration preset cases (includes enterprise showcase).
   */
  async getDemoCases() {
    const res = await fetch(`${API_BASE}/api/demo/cases`);
    if (!res.ok) {
      throw new Error(`Failed to load preset cases (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Fetch curated enterprise showcase scenarios.
   */
  async getShowcaseCases() {
    const res = await fetch(`${API_BASE}/api/showcase/cases`);
    if (!res.ok) {
      throw new Error(`Failed to load showcase cases (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Seed the live database with multi-domain verified outcomes.
   */
  async seedDatabase() {
    const res = await fetch(`${API_BASE}/api/database/seed`, {
      method: "POST"
    });
    if (!res.ok) {
      throw new Error(`Failed to seed database (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Retrieve text of a ground-truth evidence document.
   */
  async getSourceDocument(filename) {
    const res = await fetch(`${API_BASE}/api/sources/${encodeURIComponent(filename)}`);
    if (!res.ok) {
      throw new Error(`Failed to load source document (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Get configured providers and cloud presets.
   */
  async getProviders() {
    const res = await fetch(`${API_BASE}/api/providers`);
    if (!res.ok) {
      throw new Error(`Failed to load providers (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Execute task evaluation (Synthetic or Live).
   */
  async evaluateTask({ question, category = "general", options = null, mode = "synthetic", task_id = null, model = null, ensemble = false, provider = null, api_key = null, base_url = null }, signal = null) {
    const res = await fetch(`${API_BASE}/api/task/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        category,
        options,
        mode,
        task_id,
        model,
        ensemble,
        provider,
        api_key,
        base_url
      }),
      signal
    });

    if (!res.ok) {
      let detail = `Execution failed (HTTP ${res.status})`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) detail = errorJson.detail;
      } catch (e) {
        // Fallback to text status
      }
      throw new Error(detail);
    }

    return await res.json();
  }

  /**
   * Record a ground-truth outcome for a task.
   */
  async recordOutcome({ task_id, correct_answer = null, resolution_source, mode = "synthetic" }) {
    const res = await fetch(`${API_BASE}/api/outcome/record`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_id,
        correct_answer,
        resolution_source,
        mode
      })
    });

    if (!res.ok) {
      let detail = `Failed to record outcome (HTTP ${res.status})`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) detail = errorJson.detail;
      } catch (e) {}
      throw new Error(detail);
    }

    return await res.json();
  }

  /**
   * Fetch current agent trust profiles.
   */
  async getAgents(mode = "synthetic") {
    const res = await fetch(`${API_BASE}/api/agents?mode=${encodeURIComponent(mode)}`);
    if (!res.ok) {
      throw new Error(`Failed to load agent profiles (HTTP ${res.status})`);
    }
    return await res.json();
  }

  /**
   * Run the held-out synthetic benchmark comparing the 4 strategies.
   */
  async getBenchmark({ seed = 7, development_size = 300, test_size = 300 } = {}, signal = null) {
    const params = new URLSearchParams({
      seed: seed.toString(),
      development_size: development_size.toString(),
      test_size: test_size.toString()
    });
    const res = await fetch(`${API_BASE}/api/benchmark?${params.toString()}`, { signal });
    if (!res.ok) {
      let detail = `Benchmark failed (HTTP ${res.status})`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) detail = errorJson.detail;
      } catch (e) {}
      throw new Error(detail);
    }
    return await res.json();
  }
}

// Export singleton instance
window.api = new TrustNetAPI();
