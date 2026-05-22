x86_hgemm_headers = [
  'hgemm.h',
  'hgemm_common.h',
  'hgemm_noTrans.h',
  'hgemm_pack.h',
  'hgemm_util.h',
  'hgemv.h',
]

subdir('hgemm_kernel')
nntrainer_inc += include_directories('hgemm_kernel')
nntrainer_inc_abs += meson.current_source_dir() / 'hgemm_kernel'

x86_hgemm_sources = [
  'hgemm.cpp',
  'hgemm_noTrans.cpp',
  'hgemm_pack.cpp',
  'hgemm_util.cpp',
  'hgemv.cpp',
]

foreach s : x86_hgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : x86_hgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
