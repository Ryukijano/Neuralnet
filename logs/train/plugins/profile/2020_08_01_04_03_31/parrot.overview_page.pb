?	?`?
?+???`?
?+??!?`?
?+??	M]9??2@M]9??2@!M]9??2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?`?
?+??????~??A??D????Y?qS??*	?&1?dR@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat9??? ??!8??@@)?q???[??1?|V?8?=@:Preprocessing2F
Iterator::Model!???Ơ?!?\82?DF@)?????*??1?M9ϙ?:@:Preprocessing2S
Iterator::Model::ParallelMap?S?[Ɗ?!)k7???1@)?S?[Ɗ?1)k7???1@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap*?t??!??6???1@)?Yh?4??1Fݸ?ؚ*@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip?&????!r???L?K@)@??wԘp?1???T?@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::TensorSlice??z2??k?!Pti???@)??z2??k?1Pti???@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor??bc^G\?!U????@)??bc^G\?1U????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 18.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2B21.5 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????~??????~??!????~??      ??!       "      ??!       *      ??!       2	??D??????D????!??D????:      ??!       B      ??!       J	?qS???qS??!?qS??R      ??!       Z	?qS???qS??!?qS??JCPU_ONLY2black"?
both?Your program is MODERATELY input-bound because 18.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendationN
nohigh"B21.5 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 