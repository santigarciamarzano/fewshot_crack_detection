SUPPORT BRANCH
──────────────
support_img:  B × 3 × 256 × 256
      │
      ▼
 ResNetEncoder
      │
      ├── layer1: B × 64  × 64 × 64   ← no se usa en support
      ├── layer2: B × 128 × 32 × 32   ← no se usa en support
      ├── layer3: B × 256 × 16 × 16   ← no se usa en support
      └── layer4: B × 512 × 8  × 8    ← solo este importa
                      │
                      ▼
            PrototypeModule(features=layer4, mask)
                      │
                      ├── proto_crack: B × 512
                      └── proto_bg:    B × 512


QUERY BRANCH
────────────
query_img:  B × 3 × 256 × 256
      │
      ▼
 ResNetEncoder  (mismos pesos — Siamese)
      │
      ├── layer1: B × 64  × 64 × 64   ───────────────────┐
      ├── layer2: B × 128 × 32 × 32   ──────────────────┐│
      ├── layer3: B × 256 × 16 × 16   ─────────────────┐││
      └── layer4: B × 512 × 8  × 8                     │││
                      │                                │││
                      ▼                                │││
      SimilarityModule(layer4, proto_crack, proto_bg)  │││
                      │                                │││
                      └── sim_map: B × 2 × 8 × 8       │││
                                                       │││
DECODER                                                │││
───────                                                │││
cat(layer4, sim_map) → B × 514 × 8 × 8                 │││
                      │                                │││
                      ▼                                │││
              Upsample → 16 × 16  ◄── skip layer3 ─────┘││
                      │                                 ││
                      ▼                                 ││
              Upsample → 32 × 32  ◄── skip layer2 ──────┘│
                      │                                  │
                      ▼                                  │
              Upsample → 64 × 64  ◄── skip layer1 ───────┘
                      │
                      ▼
              Upsample → 128 × 128
                      │
                      ▼
              Upsample → 256 × 256
                      │
                      ▼
              Conv 1×1 → B × 1 × 256 × 256   ← máscara final
                      │
                      ▼
                   LOSS
              (solo query, nunca support)



Forward de FewShotModel
───────────────────────

forward(support_img, support_mask, query_img) → mask_logits

1. encoder(support_img) → support_features
2. encoder(query_img)   → query_features        ← mismo encoder, pesos compartidos

3. prototype(support_features["layer4"], support_mask) → proto_crack, proto_bg

4. similarity(query_features["layer4"], proto_crack, proto_bg) → sim_map

5. bottleneck = cat(query_features["layer4"], sim_map)  → B × 514 × 8 × 8

6. decoder(bottleneck, skips=query_features) → B × 1 × 256 × 256


FLUJO A ALTO NIVEL:
──────────────────

support_img  ──→ encoder ──→ PrototypeModule ──→ proto_crack
                                                  proto_bg
                                                      │
query_img    ──→ encoder ──→ layer4 ──→ SimilarityModule ──→ sim_map
                         ──→ layer1, layer2, layer3 (skips)
                                                      │
                         cat(query layer4, sim_map) ──→ UNetDecoder ──→ logits





