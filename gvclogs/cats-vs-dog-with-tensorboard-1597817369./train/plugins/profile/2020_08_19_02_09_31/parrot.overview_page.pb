?	W#??0[@W#??0[@!W#??0[@	@??C????@??C????!@??C????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$W#??0[@W[??????AԀAҧ[@YC7?????*	??????q@2F
Iterator::Model???R?1??!?.?%v?E@)??_ ???1`?;?9?=@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat?Y?X??!????9@)??I~į??1GE???7@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateV(??????!"h? ?6@) R?8?ߩ?1v?=???1@:Preprocessing2S
Iterator::Model::ParallelMap?'eRC??!b??e?*@)?'eRC??1b??e?*@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip?;???!8?fډiL@)(E+????1??)???@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??#???!??Iie?@)??#???1??Iie?@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap???I??!?CAO9@)d*??g|?1܆?w?@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorx` ?C?v?!=??/1??)x` ?C?v?1=??/1??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W[??????W[??????!W[??????      ??!       "      ??!       *      ??!       2	ԀAҧ[@ԀAҧ[@!ԀAҧ[@:      ??!       B      ??!       J	C7?????C7?????!C7?????R      ??!       Z	C7?????C7?????!C7?????JCPU_ONLY2black"?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: 