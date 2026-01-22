#pragma once

#include "skepu3/impl/common.hpp"



namespace skepu
{
    namespace impl
    {
        template<typename T>
		class MapOverlapBase
		{
		public:
			void setEdgeMode(Edge mode)
			{
				this->m_edge = mode;
			}

			Edge getEdgeMode() const
			{
				return this->m_edge;
			}

			void setPad(T pad)
			{
				this->m_pad = pad;
			}

			T getPad() const
			{
				return this->m_pad;
			}

			void setUpdateMode(UpdateMode mode)
			{
				this->m_updateMode = mode;
			}

			UpdateMode getUpdateMode() const
			{
				return this->m_updateMode;
			}

		protected:
			Edge m_edge = Edge::Duplicate;
			UpdateMode m_updateMode = skepu::UpdateMode::Normal;
			T m_pad {};
            int m_overlap[4] = {1, 1, 1, 1};
            StrideList<4> m_strides{1, 1, 1, 1};
		};
    }
}