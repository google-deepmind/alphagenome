"""MCP server for 17p11.2 locus data acquisition via AlphaGenome API.

Implements the Model Context Protocol (MCP) over stdio using raw JSON-RPC,
with no external dependencies beyond the Python standard library and the
AlphaGenome SDK.

Usage:
  python3 mcp_17p11_2_server.py

Register in Claude Code:
  claude mcp add 17p11-2-locus -- python3 /path/to/mcp_17p11_2_server.py
"""

import json
import os
import sys
import traceback

import numpy as np

from alphagenome.data import genome
from alphagenome.models import dna_client


# Default locus coordinates (hg38)
DEFAULT_CHROMOSOME = 'chr17'
DEFAULT_START = 16_700_000
DEFAULT_END = 24_100_000

# Supported sequence lengths for AlphaGenome API
SUPPORTED_LENGTHS = [16384, 131072, 524288, 1048576]

# Default tissues (UBERON ontology terms)
DEFAULT_TISSUES = [
    'UBERON:0000955',  # Brain
    'UBERON:0002048',  # Lung
    'UBERON:0002107',  # Liver
    'UBERON:0002113',  # Kidney
]

# Default output types
DEFAULT_OUTPUT_TYPES = ['RNA_SEQ', 'ATAC', 'DNASE']

OUTPUT_TYPE_MAP = {
    'RNA_SEQ': dna_client.OutputType.RNA_SEQ,
    'ATAC': dna_client.OutputType.ATAC,
    'DNASE': dna_client.OutputType.DNASE,
}


def _chunk_locus(chromosome, start, end, chunk_size=524288):
  """Break a genomic locus into API-compatible chunks."""
  chunks = []
  current_pos = start
  while current_pos < end:
    chunk_end = min(current_pos + chunk_size, end)
    chunk = genome.Interval(
        chromosome=chromosome,
        start=current_pos,
        end=chunk_end,
    )
    if chunk.width in SUPPORTED_LENGTHS:
      chunks.append(chunk)
    current_pos = chunk_end
  return chunks


def _acquire_locus_data(params):
  """Run the locus acquisition pipeline."""
  chromosome = params.get('chromosome', DEFAULT_CHROMOSOME)
  start = params.get('start', DEFAULT_START)
  end = params.get('end', DEFAULT_END)
  output_type_names = params.get('output_types', DEFAULT_OUTPUT_TYPES)
  tissues = params.get('tissues', DEFAULT_TISSUES)
  output_dir = params.get(
      'output_dir', f'{chromosome}_17p11_2_outputs_hg38'
  )

  api_key = os.getenv('ALPHAGENOME_API_KEY')
  if not api_key:
    return {
        'error': (
            'ALPHAGENOME_API_KEY environment variable not set. '
            'Please set it with your API key.'
        )
    }

  # Map output type strings to enums
  requested_outputs = []
  for name in output_type_names:
    name_upper = name.upper()
    if name_upper in OUTPUT_TYPE_MAP:
      requested_outputs.append(OUTPUT_TYPE_MAP[name_upper])
    else:
      return {'error': f'Unknown output type: {name}. Valid: {list(OUTPUT_TYPE_MAP.keys())}'}

  # Create chunks
  chunks = _chunk_locus(chromosome, start, end)
  if not chunks:
    return {
        'error': (
            f'No valid chunks for region {chromosome}:{start}-{end}. '
            f'Region must align to supported lengths: {SUPPORTED_LENGTHS}'
        )
    }

  # Connect and predict
  model = dna_client.create(api_key)

  os.makedirs(output_dir, exist_ok=True)
  results = []
  errors = []

  for i, chunk in enumerate(chunks):
    try:
      output = model.predict_interval(
          interval=chunk,
          requested_outputs=requested_outputs,
          ontology_terms=tissues,
      )

      chunk_name = (
          f'chunk_{i + 1:02d}_{chromosome}_{chunk.start}_{chunk.end}'
      )
      saved_files = []

      if output.rna_seq is not None:
        path = f'{output_dir}/{chunk_name}_rna_seq.npy'
        np.save(path, output.rna_seq.values)
        saved_files.append(path)

      if output.atac is not None and output.atac.values.shape[1] > 0:
        path = f'{output_dir}/{chunk_name}_atac.npy'
        np.save(path, output.atac.values)
        saved_files.append(path)

      if output.dnase is not None:
        path = f'{output_dir}/{chunk_name}_dnase.npy'
        np.save(path, output.dnase.values)
        saved_files.append(path)

      results.append({
          'chunk': i + 1,
          'region': f'{chromosome}:{chunk.start}-{chunk.end}',
          'files': saved_files,
      })

    except Exception as e:
      errors.append({
          'chunk': i + 1,
          'region': f'{chromosome}:{chunk.start}-{chunk.end}',
          'error': str(e),
      })

  return {
      'region': f'{chromosome}:{start}-{end}',
      'total_chunks': len(chunks),
      'successful': len(results),
      'failed': len(errors),
      'output_dir': output_dir,
      'chunks': results,
      'errors': errors,
  }


# --- MCP JSON-RPC stdio transport ---

TOOL_DEFINITIONS = [
    {
        'name': 'acquire_locus_data',
        'description': (
            'Acquire genomic predictions for a chromosomal region using '
            'the AlphaGenome API. Breaks the region into chunks, predicts '
            'RNA-seq, ATAC, and DNase tracks across tissues, and saves '
            'results as .npy files. Defaults to the 17p11.2 locus '
            '(chr17:16700000-24100000, hg38).'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'chromosome': {
                    'type': 'string',
                    'description': 'Chromosome name (e.g. chr17)',
                    'default': 'chr17',
                },
                'start': {
                    'type': 'integer',
                    'description': 'Start position (hg38)',
                    'default': 16700000,
                },
                'end': {
                    'type': 'integer',
                    'description': 'End position (hg38)',
                    'default': 24100000,
                },
                'output_types': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Track types to predict: RNA_SEQ, ATAC, DNASE'
                    ),
                    'default': ['RNA_SEQ', 'ATAC', 'DNASE'],
                },
                'tissues': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'UBERON ontology terms for tissues '
                        '(e.g. UBERON:0000955 for brain)'
                    ),
                    'default': [
                        'UBERON:0000955',
                        'UBERON:0002048',
                        'UBERON:0002107',
                        'UBERON:0002113',
                    ],
                },
                'output_dir': {
                    'type': 'string',
                    'description': 'Directory to save output files',
                },
            },
        },
    },
]

SERVER_INFO = {
    'name': '17p11-2-locus',
    'version': '1.0.0',
}


def handle_request(req):
  """Handle a single JSON-RPC request."""
  method = req.get('method', '')
  req_id = req.get('id')
  params = req.get('params', {})

  if method == 'initialize':
    return {
        'jsonrpc': '2.0',
        'id': req_id,
        'result': {
            'protocolVersion': '2024-11-05',
            'capabilities': {'tools': {'listChanged': False}},
            'serverInfo': SERVER_INFO,
        },
    }

  elif method == 'notifications/initialized':
    return None  # Notification, no response

  elif method == 'tools/list':
    return {
        'jsonrpc': '2.0',
        'id': req_id,
        'result': {'tools': TOOL_DEFINITIONS},
    }

  elif method == 'tools/call':
    tool_name = params.get('name', '')
    arguments = params.get('arguments', {})

    if tool_name == 'acquire_locus_data':
      try:
        result = _acquire_locus_data(arguments)
        return {
            'jsonrpc': '2.0',
            'id': req_id,
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps(result, indent=2),
                    }
                ],
            },
        }
      except Exception as e:
        return {
            'jsonrpc': '2.0',
            'id': req_id,
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps(
                            {
                                'error': str(e),
                                'traceback': traceback.format_exc(),
                            }
                        ),
                    }
                ],
                'isError': True,
            },
        }
    else:
      return {
          'jsonrpc': '2.0',
          'id': req_id,
          'error': {
              'code': -32601,
              'message': f'Unknown tool: {tool_name}',
          },
      }

  elif method == 'ping':
    return {'jsonrpc': '2.0', 'id': req_id, 'result': {}}

  else:
    # Unknown method — ignore notifications, error on requests
    if req_id is not None:
      return {
          'jsonrpc': '2.0',
          'id': req_id,
          'error': {
              'code': -32601,
              'message': f'Method not found: {method}',
          },
      }
    return None


def main():
  """Run the MCP server over stdio."""
  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    try:
      req = json.loads(line)
      response = handle_request(req)
      if response is not None:
        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()
    except json.JSONDecodeError:
      err = {
          'jsonrpc': '2.0',
          'id': None,
          'error': {'code': -32700, 'message': 'Parse error'},
      }
      sys.stdout.write(json.dumps(err) + '\n')
      sys.stdout.flush()


if __name__ == '__main__':
  main()
