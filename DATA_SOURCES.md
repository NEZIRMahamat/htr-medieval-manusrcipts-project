# CATMuS HTR Training Data Notes

Dataset: `CATMuS/medieval`
Config/subset: `default`
Task: image-to-text handwritten text recognition.

Source used for this audit:
- Hugging Face dataset metadata: `https://datasets-server.huggingface.co/info?dataset=CATMuS/medieval`
- Hugging Face first rows: `https://datasets-server.huggingface.co/first-rows?dataset=CATMuS/medieval&config=default&split=train`

## Splits and Counts

| Split | Samples |
| --- | ---: |
| `train` | 152,816 |
| `validation` | 19,402 |
| `test` | 22,590 |
| **Total** | **194,808** |

## Fields

Image field: `im`
Transcription field: `text`
Metadata fields: `language, century, region, script_type, shelfmark, verse, genre, project, line_type, gen_split`

For the HTR part, the model input is the line image in `im`, and the target text is `text`.

## Feature Schema

```json
{
  "text": {
    "dtype": "string",
    "_type": "Value"
  },
  "im": {
    "_type": "Image"
  },
  "language": {
    "dtype": "string",
    "_type": "Value"
  },
  "century": {
    "dtype": "int8",
    "_type": "Value"
  },
  "region": {
    "dtype": "string",
    "_type": "Value"
  },
  "script_type": {
    "dtype": "string",
    "_type": "Value"
  },
  "shelfmark": {
    "dtype": "string",
    "_type": "Value"
  },
  "verse": {
    "dtype": "string",
    "_type": "Value"
  },
  "genre": {
    "dtype": "string",
    "_type": "Value"
  },
  "project": {
    "dtype": "string",
    "_type": "Value"
  },
  "line_type": {
    "dtype": "string",
    "_type": "Value"
  },
  "gen_split": {
    "dtype": "string",
    "_type": "Value"
  }
}
```

## One Training Sample

```json
{
  "text": "q̃  y paresçio. ffecha ⁊ signada por man de domĩgo ffeĩrs escrͥuan publico de Burgos del rregistͦ de pͦ m̃z escͥuan. ffizierõ abenẽçia en esta maña q̃los dich̃s Sanch̃ ꝑez ⁊",
  "im": {
    "type": "image",
    "height": 170,
    "width": 4772,
    "src_without_temporary_signature": "https://datasets-server.huggingface.co/assets/CATMuS/medieval/--/e11965909ba89dea89476f665fc4d8541b0bf7a1/--/default/train/0/im/image.jpg"
  },
  "language": "Castilian",
  "century": 13,
  "region": "MainZone",
  "script_type": "Semihybrida",
  "shelfmark": "Paris, BnF, esp. 480",
  "verse": "prose",
  "genre": "Documents of practice",
  "project": "HTRomance",
  "line_type": "DefaultLine",
  "gen_split": "test"
}
```

## Report Notes

- CATMuS/medieval is already line-level HTR data: each row contains one text-line image and its transcription.
- The line image column is `im`; the transcription/label column is `text`.
- Useful metadata to preserve in JSON outputs: `language`, `century`, `region`, `script_type`, `shelfmark`, `verse`, `genre`, `project`, `line_type`, and `gen_split`.
- The official Hugging Face split names are `train`, `validation`, and `test`.
- The sample rows also contain `gen_split`, whose values can be `train`, `dev`, or `test`; keep it as metadata, but use the official split for training/evaluation separation.
- Keep the `test` split sealed until final evaluation to avoid data leakage.
