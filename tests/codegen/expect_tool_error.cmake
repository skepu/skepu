# Runs skepu-tool on a source and checks what it does:
#   -DEXPECT=text   it must refuse the source: exit status 1 (not 0, and not a crash) and the text in what it says
#   no EXPECT       it must accept the source (exit status 0)
# Arguments: TOOL (skepu-tool), SOURCE, OUT (a directory for what it makes), FLAGS (the flags after --, separated by |), EXPECT.
string(REPLACE "|" ";" _flags "${FLAGS}")
file(REMOVE_RECURSE "${OUT}")
file(MAKE_DIRECTORY "${OUT}")

execute_process(
	COMMAND "${TOOL}" -silent -name tool_error_test -dir=${OUT} "${SOURCE}" -- ${_flags}
	RESULT_VARIABLE _status
	OUTPUT_VARIABLE _output
	ERROR_VARIABLE _errors)

if(EXPECT)
	if(NOT "${_status}" STREQUAL "1")
		message(FATAL_ERROR "skepu-tool was to refuse ${SOURCE} with the status 1, and ended with '${_status}'.\n${_output}\n${_errors}")
	endif()
	if(NOT "${_output}${_errors}" MATCHES "${EXPECT}")
		message(FATAL_ERROR "skepu-tool refused ${SOURCE}, but not with a message that has '${EXPECT}':\n${_output}\n${_errors}")
	endif()
else()
	if(NOT "${_status}" STREQUAL "0")
		message(FATAL_ERROR "skepu-tool was to accept ${SOURCE}, and ended with '${_status}'.\n${_output}\n${_errors}")
	endif()
endif()
