"""Generates C4_200M sentence pairs from edits and C4 sentences."""
import json

from absl import app


def get_edits(edits_tsv_reader):
  """Generator method for the edits."""
  current_edit_md5 = None
  for edits_tsv_line in edits_tsv_reader:
    try:
      edit_md5, byte_start, byte_end, text = edits_tsv_line.strip("\n").split(
          "\t", 3)
    except ValueError:
      pass  # Skip malformed lines
    else:
      if edit_md5 != current_edit_md5:
        if current_edit_md5 is not None:
          yield current_edit_md5, current_edits
        current_edit_md5 = edit_md5
        current_edits = []

      text = str(text)
      words = text.strip().split(' ')
      if len(words) >= 3:
        continue

      n_valid_words = sum([x.replace("'", '').isalpha() for x in words])
      if n_valid_words != len(words):
        continue

      current_edits.append((int(byte_start), int(byte_end), text))
  yield current_edit_md5, current_edits


def apply_edits(edits, target_text):
  target_bytes = target_text.encode("utf-8")
  last_byte_position = 0
  source_text = ""

  updates = []
  for byte_start, byte_end, replacement_text in edits:
    source_text += target_bytes[last_byte_position:byte_start].decode("utf-8")
    source_text += replacement_text
    last_byte_position = byte_end

    if not target_bytes[:byte_start].decode("utf-8").strip():
      b = 0
    else:
      b = len(target_bytes[:byte_start].decode("utf-8").strip().split(' '))
    updates.append({
      'rp_begin': b,
      'rp_length': len(target_bytes[byte_start:byte_end].decode("utf-8").strip().split(' ')),
      'rp_from': target_bytes[byte_start:byte_end].decode("utf-8").strip(),
      'rp_to': replacement_text.strip()
    })

  source_text += target_bytes[last_byte_position:].decode("utf-8")
  return source_text, updates


def main(argv):
  if len(argv) != 4:
    raise app.UsageError("python3 c4200m_make_sentence_pairs.py "
                         "<target-sentence-tsv> <edits-tsv> <output-tsv>")
  target_sentence_tsv_path = argv[1]
  edits_tsv_path = argv[2]
  output_tsv_path = argv[3]

  with open(edits_tsv_path) as edits_tsv_reader:
    with open(target_sentence_tsv_path) as target_sentence_tsv_reader:
      with open(output_tsv_path, "w") as output_tsv_writer:
        edits_iterator = get_edits(edits_tsv_reader)
        edit_md5 = "0"
        for target_sentence_tsv_line in target_sentence_tsv_reader:
          md5, target_text = target_sentence_tsv_line.strip("\n").split("\t", 1)
          while edit_md5 < md5:
            edit_md5, edits = next(edits_iterator)
          if edit_md5 == md5:
            source_text, updates = apply_edits(edits, target_text)
            if not updates:
              continue
            output_tsv_writer.write("%s\t%s\t%s\n" % (source_text, target_text, json.dumps(updates)))


if __name__ == "__main__":
  app.run(main)
