import streamlit as st
import requests, json, os, platform, zlib
from streamlit.runtime.scriptrunner import get_script_run_ctx


def main():
  ctx = get_script_run_ctx()

  st.title('Ollama test')

  print('ISID',ctx.session_id)

  if 'SID' not in st.session_state:
    st.session_state['SID']=ctx.session_id
    # if platform.system() == "Windows":
    #   pcID=platform.uname().node
    # else:
    #   pcID=os.uname()[1]

    # st.session_state['SID'] = str(zlib.crc32(pcID.encode()))
    # print('pcID', pcID)
  else:
    print(' SID',st.session_state.SID)

  s_id=st.session_state.SID

  st.subheader("Ollama Chat", divider="green", anchor=False)

  message_container = st.container(height=500, border=True)

  if "messages" not in st.session_state:
        st.session_state.messages = []

  for message in st.session_state.messages:
      print('',end='.')
      avatar = "🤖" if message["role"] == "assistant" else "😎"
      with message_container.chat_message(message["role"], avatar=avatar):
          st.markdown(message["content"])
  print('')

  if prompt := st.chat_input("Enter a prompt here..."):
    try:
      st.session_state.messages.append(
        {"role": "user", "content": prompt}
      )

      message_container.chat_message("user", avatar="😎").markdown(prompt)

      # . . . ~ ~
      print(f"http://127.0.01:5000/api?query={prompt}&sid={s_id}")
      req=requests.get(f"http://127.0.01:5000/api?query={prompt}&sid={s_id}", stream=True)
      print(req.status_code)
      if req.status_code==200:
      # for line in r.iter_lines():
      #   if line:
      #       print(line)
        for line in req.iter_lines():
           if line:
            print(type(line), line)

        dati=json.loads(req.text)
        testo=f"{dati['answer']}\n\n*query in: {dati['time']:.02f}sec*"
        print('>>>',testo)
        st.session_state.messages.append(
          {"role": "assistant", "content": testo}
        )
        message_container.chat_message("assistant", avatar="🤖").markdown(testo)
    except Exception as e:
      st.error(e, icon="⛔️")



st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if __name__ == "__main__":
    main()
