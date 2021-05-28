# grapheneABC
This program parametrically generates CSV files that represented laser-scanned transferred graphene. The program then uses a rejection ABC method to update those parameters based on an experimental CSV file.

The full_code.py file is meant to be run in the Spyder IDE (https://www.spyder-ide.org/); although, any python IDE should suffice.
	All functions are found in this file including:
	- generating graphene (stored in "Generation" folder)
	- scanning graphene using a simulated laser (stored in "Scanning" folder)
	- ABC rejection of graphene models to experimental graphene

In order for it to work:
	- input parameters must be present in input.csv in Generation folder
	- File pathway must be correct in "full_code.py"
	- experimental.csv must be present in Scanning folder.

