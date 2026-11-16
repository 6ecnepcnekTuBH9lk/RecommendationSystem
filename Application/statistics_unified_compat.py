"""Source hashes of verified, output-equivalent Unified implementations."""

# Each alias applies only to the exact reviewed implementation. Further edits
# still invalidate the appropriate layer through its normal content hash.
_EQUIVALENT = {
    ("statistics_unified_aggregate_input.py", "e54372dfe8c0a100fdf5eea2c8036ca00857e67734b73be56cf6dacf36dbda0f"):
        "c734d755c202133c07b193f89f55016820440f53311dbe97536b450df44569f7",
    ("statistics_unified_analytics.py", "1e9fe177f58d8bbdfa3a28c31627392930725cb458b7faabc52ea82baa340346"):
        "530c788b45b18fe3674a08878d6eff7a0900ae482fb9afda530273bbb4e496f3",
    ("statistics_unified_aggregates.py", "182f8fc1061cb0591f865e420dfac375c0635d2b5f5c934cae6cbd1605148993"):
        "dca617be679eae5b9459d858e1fc5f4411e48b23fc1f689fd39f71f9e25e3bbe",
    ("statistics_unified_service.py", "26c2b5ca9f7aa2c02f1768853740a350897e1be483b9dd2ffce1013375356d85"):
        "dad11f91422a15fce6ab3fc51fc76d295c3e09106cf57b4423c0ff1fdb14c74f",
    ("statistics_unified_analytics.py", "02c5611f55634948a7a653f514930e31102434a210eb1114ec49bcbd7b507ecf"):
        "530c788b45b18fe3674a08878d6eff7a0900ae482fb9afda530273bbb4e496f3",
    ("statistics_unified_aggregates.py", "3cf20c20134cc9a9cf24e1773e8e9e6f4a590e9953e23f528dccc306623c8e6e"):
        "dca617be679eae5b9459d858e1fc5f4411e48b23fc1f689fd39f71f9e25e3bbe",
    ("statistics_unified_breakdowns.py", "fe875e59727720f85f90f59c10e2b84291486a79ba88b4ffc4426f099442cab8"):
        "04be4d5fe197a6ec77ae1be18bbb4243a0f72c2822c91b39ba1fa6ae935e74b1",
    ("statistics_unified_service.py", "409de71e08879ffd14a29d69ba1a5b8561aada07a49b8d27324dbf202c874f9f"):
        "dad11f91422a15fce6ab3fc51fc76d295c3e09106cf57b4423c0ff1fdb14c74f",
    # Remove an unused local from _money_payload after publishing V2.
    ("statistics_unified_cohorts.py", "68b3a6ca3ea530bb257b06ca5fce1c4dbd021d9873fc0c7437b108af7af761ae"):
        "f43a08406463ec3204a6bec13c9e9951e5522d9ccfedad86c0b838bdeaec9e8a",
}


def equivalent_hash(name, digest):
    return _EQUIVALENT.get((name, digest), digest)


# These revisions preserve the on-disk reconciled facts and population.
# Readers check the segmentation policy separately before reusing an RFM summary.
# Result fingerprints still hash the current implementation.
_PREPARED_FACTS_COMPATIBILITY = {
    ("statistics_unified_rfm.py", "8fc75caf9c1d07a826fc52e07ac3bcb8ee4e923a930d86543b7e6cb6fa561823"):
        "013b2dd992a46c70d7de90657f05e602ffb330aea8b42f55b5d0248911f7b0b9",
    ("statistics_unified_analytics.py", "f7e5d6747b9904a963d9b90bd8d6afff17e6c65612d52bc2e9a03f37d8ccb6ff"):
        "85c08de5c72a98c507ae4f2e1a4ec7cbfffe85c72b69f17dbe73c1c50ba24568",
}


def prepared_component_hash(name, digest):
    return equivalent_hash(name, _PREPARED_FACTS_COMPATIBILITY.get((name, digest), digest))
