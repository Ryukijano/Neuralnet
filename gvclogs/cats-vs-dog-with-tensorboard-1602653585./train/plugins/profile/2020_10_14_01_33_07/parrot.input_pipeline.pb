	Bx?q?b@Bx?q?b@!Bx?q?b@	???z???????z????!???z????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Bx?q?b@ʦ\?]??A A??gb@Yz?9[@h??*	\d;?O?x@2F
Iterator::Model.Ȗ??2??!mg??,?C@)??R??F??1읧zh?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatap??/??!?LC??>@)8?ܘ????1,v%'?&8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateb?G??!?E?oS?3@)??2p@??1fhoS?*@:Preprocessing2U
Iterator::Model::ParallelMapV2????=??!?a?e??!@)????=??1?a?e??!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????c???!??4)?=N@)J????i??1G?>?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?T?z???!Zw?gl@)?T?z???1Zw?gl@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice܂??????!?F:??@)܂??????1?F:??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_$??\???!?Rm??7@)@??$"??1?f8??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???z????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ʦ\?]??ʦ\?]??!ʦ\?]??      ??!       "      ??!       *      ??!       2	 A??gb@ A??gb@! A??gb@:      ??!       B      ??!       J	z?9[@h??z?9[@h??!z?9[@h??R      ??!       Z	z?9[@h??z?9[@h??!z?9[@h??JCPU_ONLYY???z????b 