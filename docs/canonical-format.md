# Canonical SGLang KV format v1

## Page key

Input tokens are split into page-size groups. Each key is chained SHA-256:

```text
key[0] = sha256(u32le(tokens[0:page_size]))
key[n] = sha256(key[n-1].bytes || u32le(tokens[n*page_size:(n+1)*page_size]))
```

This matches ordinary SGLang token-page keys. Prefix lengths must be page
aligned. Generated output pages are not part of a Decode-prefix checkpoint.

## Canonical page

For MHA/GQA, a canonical page is a byte-exact view with axes:

```text
[K_or_V, token_in_page, global_layer, global_kv_head, head_dim, item_byte]
```

Its shape is:

```text
[2, page_size, num_layers, num_kv_heads, head_dim, dtype_item_size]
```

The final byte axis lets the converter preserve `float16`, `bfloat16`, or
`float32` payloads without numeric conversion.

Files live at:

```text
<disk_root>/<canonical_fingerprint>/pages/<key[0:2]>/<key>.bin
```

`manifest.json` records format identity, prefixes, page count, page byte size,
and production provenance.

## SGLang materialization

The current adapter reads and writes SGLang `HiCacheFile` page-first rank files.
A rank file has the byte-view shape:

```text
[2, page_size, local_layers, local_kv_heads, head_dim, dtype_item_size]
```

For TP, the canonical KV-head axis is sliced by `tp_rank`. For PP, the canonical
layer axis is sliced by the explicit `layer_partition`. Combined TP+PP applies
both slices. Filenames follow SGLang's suffix convention:

```text
<page_key>_<model>_<tp_rank>_<tp_size>[_<pp_size>_<pp_rank>].bin
```

## Canonical compatibility

The canonical fingerprint contains:

- canonical format and page-key algorithm versions;
- SGLang engine version/source revision;
- model identity, revision, configuration and index digests;
- dtype, quantization, and full KV layout;
- ordered prefix token digests and lengths.

It excludes topology, TP/PP sizes, node names, endpoint routing, checkpoint IDs,
and local paths. Those belong to the separate materialization fingerprint.

## Unsupported layouts

- MLA;
- non-page-first HiCache files;
- TP that does not divide `num_key_value_heads`;
- PP without a complete explicit layer partition;
- heterogeneous rank counts within one instance's node list.

Unsupported cases fail before any data is published or installed.
