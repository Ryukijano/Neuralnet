	?Aȗ?Y@?Aȗ?Y@!?Aȗ?Y@	??!! ?????!! ???!??!! ???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Aȗ?Y@?D?????A> Й??Y@Y????k???*	?? ?rhw@2F
Iterator::Modelz?΅???!?ϋ'?GA@)'?????1???g?8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?+?,???!?}s?-?@)	?3????1?d@?u?8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?3?/.U??!?9???k:@)?j{???1+[??y?5@:Preprocessing2U
Iterator::Model::ParallelMapV2?Ϲ????!??)py?#@)?Ϲ????1??)py?#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorV??#)??!S???=@)V??#)??1S???=@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipF^??_??!&:?\P@)???c?3??1?~4?(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicezrM??Β?!Pz?t??@)zrM??Β?1Pz?t??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? ??	L??!$^Jax<@)???@?m?1kR?c @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??!! ???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?D??????D?????!?D?????      ??!       "      ??!       *      ??!       2	> Й??Y@> Й??Y@!> Й??Y@:      ??!       B      ??!       J	????k???????k???!????k???R      ??!       Z	????k???????k???!????k???JCPU_ONLYY??!! ???b 