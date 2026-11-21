import numpy as np
import pytest

from core.local_embed import LocalBgeSmallEmbeddings, find_model_dir
from core.mock_embed import MockEmbeddings

needs_model = pytest.mark.skipif(find_model_dir() is None, reason="bge-small not in local HF cache")


def test_mock_is_deterministic_normalised_and_lexical():
    m = MockEmbeddings()
    a, b, c = m.embed_documents(["input tax credit invoice", "input tax credit invoice", "penalty for kyc breach"])
    assert a == b and abs(np.linalg.norm(a) - 1) < 1e-9
    assert np.dot(a, a) > np.dot(a, c)


@needs_model
def test_local_bge_small_is_semantic_and_normalised():
    e = LocalBgeSmallEmbeddings()
    v = np.array(e.embed_documents([
        "Input tax credit is restricted to invoices reflected in the GST return.",
        "Buyers can only claim ITC for invoices that appear in their GSTR-2B.",
        "The cat sat on the warm windowsill all afternoon."]))
    assert v.shape == (3, 384) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2] + 0.2


@needs_model
def test_local_embedding_truncates_long_input_instead_of_failing():
    assert len(LocalBgeSmallEmbeddings().embed_query("tax " * 2000)) == 384
