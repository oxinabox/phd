addprocs([
          ("cluster_c4_1",:auto),
          ("cluster_c4_2",:auto),
          ("cluster_c4_3",:auto),
          ("cluster_c4_4",:auto),
          ("cluster_c4_5",:auto),
    ],
	dir="/mnt/",
	topology=:master_slave
	)

addprocs(10,
	dir="/mnt/",
	topology=:master_slave
	)

workers()
