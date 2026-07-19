# The x86 hgemm headers are internal (nntrainer::hgemm::internal) and are used
# only through the cpu_backend dispatch inside this library, so they are added to
# the include path below but intentionally not installed.


x86_hgemm_sources = [
  'hgemm_pack.cpp',
  'hgemm_util.cpp',
]

foreach s : x86_hgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
