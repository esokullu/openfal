/**
 * Strata-DiT Engine Telemetry, Hardware Profiles, Model Inventory & Sleep/Wake Control test.
 *
 * Usage:
 *   node engine_control.mjs
 */
import { OPENFAL_URL } from "./config.mjs";

async function getJson(route, method = "GET", body = undefined) {
  const res = await fetch(`${OPENFAL_URL}${route}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new Error(`${method} ${route} failed (${res.status}): ${await res.text()}`);
  }
  return res.json();
}

console.log(`=== [Engine Control] Inspecting ${OPENFAL_URL} ===`);

const health = await getJson("/health");
console.log("\n1. /health:", health);

const status = await getJson("/v1/engine/status");
console.log("\n2. /v1/engine/status:", {
  profile_id: status.profile?.profile_id ?? status.profile_id,
  active_partition: status.active_partition,
  vram_counter: status.tiers?.tier1_vram_counter,
  ram_pantry: status.tiers?.tier2_ram_pantry,
});

const hw = await getJson("/v1/models/hardware-detect");
console.log("\n3. /v1/models/hardware-detect:", {
  platform: hw.platform,
  selected_compute_gpus: hw.selected_compute_gpus,
  ignored_gpus: hw.ignored_gpus,
  recommended_profile_id: hw.recommended_profile_id,
  system_ram_gb: hw.system_ram_gb,
  free_nvme_gb: hw.free_nvme_gb,
});

const models = await getJson("/v1/models/status");
console.log("\n4. Installed model components:");
for (const c of models.components || []) {
  console.log(
    `   - ${c.component_id.padEnd(16)} : installed=${c.installed} (${c.disk_size_mb} MB)`
  );
}

const sleepRes = await getJson("/v1/engine/sleep", "POST");
console.log("\n5. POST /v1/engine/sleep:", sleepRes.status);

const wakeRes = await getJson("/v1/engine/wake", "POST");
console.log("6. POST /v1/engine/wake :", wakeRes.status);
