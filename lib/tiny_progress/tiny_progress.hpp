#ifndef TINY_PROGRESS_HPP
#define TINY_PROGRESS_HPP


#include <string>
#include <thread>
#include <fstream>
#include <iomanip>


namespace tiny_progress
{
    class ProgressBar
    {
    public:
        ProgressBar() = default;
        ProgressBar(unsigned int nTasks) : m_nTasks(nTasks) { }

        void begin(std::ostream& os, std::string sMsg) 
        {
            m_nCurTasks = 0;
            m_sMsg = sMsg;
            m_t = std::thread(load, std::ref(os), std::ref(m_sMsg), std::ref(m_nCurTasks), m_nBarWidth, m_nTasks, m_sLoadSymbol);
        }

        void update(unsigned int nUpdatedTasks, std::string sMsg)
        {
            m_sMsg = sMsg;
            m_nCurTasks += nUpdatedTasks;
        }

        void end(std::ostream& os, std::string sMsg)
        {
            bool bIsNotFinished = false;
            if(m_nCurTasks != m_nTasks)
            {
                m_nCurTasks = m_nTasks;
                bIsNotFinished = true;
            }
                
            if(m_t.joinable())
                m_t.join();
            os << "[";
            for(auto i = 0; i < m_nBarWidth - 1; ++i)
                os << "=";
            os << ">";
            os << "] 100% " << sMsg;
            for(auto i = sMsg.size(); i < m_sMsg.size() + 2; ++i)
                os << " ";
            os << "\n";
            os.flush();

            if(bIsNotFinished)
            {
                os << "[WARNING] Code not completed.\n";
                os.flush();
            }
        }
        

    private:
        static void load(std::ostream& os, std::string& sMsg, unsigned int& nCurTasks, unsigned int nBarWidth, unsigned int nTasks, std::string sLoadSymbol[4])
        {
            unsigned int nCurPos, cnt = 0;
            while(true)
            {
                os << "[";
                nCurPos = nBarWidth * (double)nCurTasks / nTasks;
                for(auto i = 0; i < nBarWidth; ++i)
                {
                    if(i < nCurPos) os << "=";
                    else if(i == nCurPos) os << ">";
                    else os << " "; 
                }
                os << "] " << std::setw(3) << std::fixed << std::setprecision(0) << (double)nCurTasks / nTasks * 100 << "% ";
                os << sMsg << " " << sLoadSymbol[(cnt / 1000) % 4] << "\r";
                os.flush();
                if(nCurTasks == nTasks)
                    break;
                cnt++;
            }
        }

        std::thread m_t;
        unsigned int m_nTasks;
        unsigned int m_nCurTasks;
        std::string m_sMsg;
        unsigned int m_nBarWidth = 70;
        std::string m_sLoadSymbol[4] = {"/", "-", "\\", "|"};
    };

};


#endif 