"""Optional Streamlit dashboard for claim-level verification results."""

import html
import os

import streamlit as st

from .nli import NliClassifier, load_label_map
from .pipeline import HallucinationVerifier


@st.cache_resource
def get_verifier(model_name: str, label_map_json: str | None) -> HallucinationVerifier:
    return HallucinationVerifier(
        NliClassifier(model_name, label_map=load_label_map(label_map_json))
    )


def main() -> None:
    st.set_page_config(page_title="Hallu Detect", page_icon="🔎", layout="wide")
    st.title("Hallu Detect")
    st.caption("Check generated claims against evidence you provide.")

    model_name = os.getenv("HALLU_MODEL")
    if not model_name:
        st.error("Set HALLU_MODEL to a fine-tuned three-class NLI checkpoint, then restart the app.")
        st.stop()

    response = st.text_area("Generated response", height=180)
    source_text = st.text_area(
        "Source documents",
        height=220,
        help="Separate documents with a blank line. Sources are processed locally by this app.",
    )
    if st.button("Verify response", type="primary"):
        if not response.strip():
            st.warning("Enter a response to verify.")
            return
        documents = [part.strip() for part in source_text.split("\n\n") if part.strip()]
        verifier = get_verifier(model_name, os.getenv("HALLU_LABEL_MAP"))
        result = verifier.verify(response, documents)

        metric_column, supported_column, contradicted_column, unknown_column = st.columns(4)
        metric_column.metric("Faithfulness ratio", f"{result['overall_faithfulness']:.1%}")
        supported_column.metric("Supported", result["supported_count"])
        contradicted_column.metric("Contradicted", result["contradicted_count"])
        unknown_column.metric("Unknown", result["unknown_count"])

        highlights_tab, evidence_tab, json_tab = st.tabs(
            ["Claim highlights", "Evidence", "Raw JSON"]
        )
        with highlights_tab:
            colors = {
                "supported": "#d1fae5",
                "contradicted": "#fee2e2",
                "unknown": "#fef3c7",
            }
            if not result["claims"]:
                st.info("No sentence-level claims were extracted.")
            for item in result["claims"]:
                color = colors[item["verdict"]]
                claim = html.escape(item["claim"])
                verdict = html.escape(item["verdict"].title())
                st.markdown(
                    f'<div style="background:{color};padding:12px;margin:8px 0;'
                    f'border-radius:6px"><strong>{verdict}</strong> '
                    f'({item["confidence"]:.1%})<br>{claim}</div>',
                    unsafe_allow_html=True,
                )
        with evidence_tab:
            for index, item in enumerate(result["claims"], start=1):
                with st.expander(f"Claim {index}: {item['claim']}"):
                    if not item["evidence"]:
                        st.info("No evidence was retrieved; verdict is Unknown.")
                    for passage in item["evidence"]:
                        st.caption(f"{passage['source']} — retrieval score {passage['score']:.3f}")
                        st.write(passage["text"])
        with json_tab:
            st.json(result)


if __name__ == "__main__":
    main()
