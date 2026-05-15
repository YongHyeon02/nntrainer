x86_hgemm_kernel_headers = [
  'hgemm_kernel.h',
]

x86_hgemm_kernel_sources = [
  'hgemm_kernel_6x16.cpp',
]

foreach s : x86_hgemm_kernel_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : x86_hgemm_kernel_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
