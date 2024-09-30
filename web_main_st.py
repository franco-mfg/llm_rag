import streamlit as st
# import ptvsd
import requests, os, platform, zlib, json
# import simplejson as json
from streamlit.runtime.scriptrunner import get_script_run_ctx

# __DEBUG__=False


def dbg(*args):
  # global __DEBUG__
  if not globals().get('__DEBUG__', True):
     return None

  res=''
  for i in args:
    res+=f' {i}'

  print(res)



def stream_server_data(prompt, sid):
  req=requests.get(f"http://127.0.01:5000/api?query={prompt}&sid={sid}", stream=True)
  if req.status_code==200:
    for line in req.iter_lines():
        if line:
          dbg(type(line), line)
          # bytes_to_str=str(line,'utf-8')
          # print(type(dataLine), dataLine)
          dbg('qui')
          jsData=json.loads(line)
          dbg('qua')
          dataLine=''
          for key in jsData:
            dbg('Key',key)
            match key:
              case 'answer':
                dataLine+=dataLine+jsData[key]
              case 'time':
                tm=jsData[key]
                dbg(type(tm),tm)
                dbg('time '+f"\n\n*query in: {float(tm):.02f}sec*")
                dataLine+=dataLine+f"\n\n*query in: {float(tm):.02f}sec*"
                dbg("dataline:",dataLine)
          yield dataLine # str(dataLine,'utf-8')

def main():
  ctx = get_script_run_ctx()

  st.title('Ollama test')

  dbg('ISID',ctx.session_id)

  if 'SID' not in st.session_state:
    st.session_state['SID']=ctx.session_id
  else:
    dbg(' SID',st.session_state.SID)

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

      response=''
      with message_container.chat_message("assistant", avatar="🤖"):
         response = st.write_stream(stream_server_data(prompt, s_id))
         dbg(f"...{response}")
      print(response)

      st.session_state.messages.append(
        {"role": "assistant", "content": response}
      )
    except Exception as e:
      st.error(e, icon="⛔️")



st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if __name__ == "__main__":
    # ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    # ptvsd.wait_for_attach()
    # __DEBUG__=False
    main()
